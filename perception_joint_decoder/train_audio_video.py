import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.checkpoints import Checkpointer

from scipy.stats import pearsonr

from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
import librosa
import PIL


class SER(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        sig, lens = batch.sig
        images = batch.images

        labels = batch.label_required
        labels = torch.tensor(labels).to(self.device)
        labels = labels.squeeze(-1)

        # Extract features from Wav2vec2
        wav2vec2_outputs = self.modules.wav2vec2(sig) 
        wav2vec2_outputs = wav2vec2_outputs[-1]

        # Extract features from the vision transformer model
        vit_feat = [self.modules.vit_model(img) for img in images]
        # zero-pad the vit_feat
        max_len = max([elem.size(0) for elem in vit_feat])
        padded_vit_feat = []
        vit_lens = []
        for feat in vit_feat:
            n = feat.size(0)
            vit_lens.append(n)
            if n < max_len:
                pad = (0, 0, 0, max_len - n)
                padded_tensor = torch.nn.functional.pad(feat, pad, "constant", 0)
            else:
                padded_tensor = feat
            padded_vit_feat.append(padded_tensor)
        vit_feat = torch.stack(padded_vit_feat, dim=0).to(self.device)
        # convert vit_lens to tensor
        vit_lens = torch.tensor(vit_lens).to(self.device)
        # take the first/last 20 images
        # vit_feat = vit_feat[:, -20:, :]
        # vit_lens = vit_lens[-20:]
        #vit_feat = vit_feat[:, :20, :]
        #vit_lens = vit_lens[:20]

        ### Autoregressive predictions ###
        output_preds_1 = []
        output_preds_2 = []
        output_preds_3 = []
        output_preds = []

        next_input = torch.zeros(wav2vec2_outputs.size(0), 1, 1).to(self.device)
        for i in range(8):
            w2v2_dec_outputs, hidden_w2v2 = self.modules.gru_decoder_1(next_input, wav2vec2_outputs, lens)
            vit_dev_outputs, hidden_vit = self.modules.gru_decoder_2(next_input, vit_feat, vit_lens)

            # Modality attention
            wav2vec2_scores = self.modules.lin_att_audio(w2v2_dec_outputs)
            vit_scores = self.modules.lin_att_video(vit_dev_outputs)

            wav2vec2_weights = torch.softmax(wav2vec2_scores, dim=1)
            vit_weights = torch.softmax(vit_scores, dim=1)

            wav2vec2_context = wav2vec2_weights * w2v2_dec_outputs
            vit_context = vit_weights * vit_dev_outputs

            # mask the modality
            # wav2vec2_context = torch.zeros_like(wav2vec2_context)

            outputs = torch.cat((wav2vec2_context, vit_context), dim=-1)
            outputs = torch.sigmoid(self.modules.output_layer_1(outputs))

            output_preds_1.append(outputs.squeeze(-1))
            next_input = outputs

        next_input = torch.zeros(wav2vec2_outputs.size(0), 1, 1).to(self.device)
        for i in range(8):
            w2v2_dec_outputs, hidden_w2v2 = self.modules.gru_decoder_1(next_input, wav2vec2_outputs, lens)
            vit_dev_outputs, hidden_vit = self.modules.gru_decoder_2(next_input, vit_feat, vit_lens)

            # Modality attention
            wav2vec2_scores = self.modules.lin_att_audio(w2v2_dec_outputs)
            vit_scores = self.modules.lin_att_video(vit_dev_outputs)

            wav2vec2_weights = torch.softmax(wav2vec2_scores, dim=1)
            vit_weights = torch.softmax(vit_scores, dim=1)

            wav2vec2_context = wav2vec2_weights * w2v2_dec_outputs
            vit_context = vit_weights * vit_dev_outputs

            # mask the modality
            # wav2vec2_context = torch.zeros_like(wav2vec2_context)

            outputs = torch.cat((wav2vec2_context, vit_context), dim=-1)
            outputs = torch.sigmoid(self.modules.output_layer_1(outputs))

            output_preds_2.append(outputs.squeeze(-1))
            next_input = outputs
        
        output_preds_1 = torch.stack(output_preds_1, dim=1).squeeze(-1)
        output_preds_2 = torch.stack(output_preds_2, dim=1).squeeze(-1)
        decoder_outputs = torch.cat((output_preds_1, output_preds_2), dim=1)

        '''
        from paper:
        aggressive, arrogant, assertive, confident, dominant, independent, risk-taking, leader-like, collaborative, enthusiastic, friendly, good-natured, kind, likeable, sincere, warm
        from labels:
        subj_id,aggressive,arrogant,dominant,enthusiastic,friendly,leader_like,likeable,assertiv,confident,independent,risk,sincere,collaborative,kind,warm,good_natured
        '''

        indexed = [0, 1, 4, 9, 10, 7, 13, 2, 3, 5, 6, 14, 8, 12, 15, 11]
        decoder_outputs = decoder_outputs[:, indexed]
        # decoder_outputs = torch.stack(output_preds, dim=1).squeeze(-1)

        wav2vec2_outputs = self.hparams.avg_pool(wav2vec2_outputs, lens).squeeze(1)
        vit_feat = self.hparams.avg_pool(vit_feat, vit_lens).squeeze(1)

        return decoder_outputs, wav2vec2_outputs, vit_feat
    

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label_required
        # Convert labels to tensor
        labels = torch.tensor(labels).to(self.device)
        decoder_outputs, wav2vec2_outputs, vit_feat = predictions

        loss = self.hparams.compute_cost(decoder_outputs, labels)
        # loss_l1 = self.hparams.l1_cost(wav2vec2_outputs, vit_feat)

        if stage != sb.Stage.TRAIN:
            decoder_outputs = decoder_outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            self.error_metrics.append((decoder_outputs, labels))

        return loss


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = []
    

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            # self.error_metrics = sum(self.error_metrics) / len(self.error_metrics)
            all_preds = []
            all_labels = []
            for elem in self.error_metrics:
                predictions, labels = elem
                all_preds.append(predictions)
                all_labels.append(labels)
            
            # stack them
            predictions = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            mean_r = np.array([pearsonr(predictions[:, i], labels[:, i])[0] for i in range(predictions.shape[1])])
            self.error_metrics = np.mean(mean_r)
            stats = {
                "loss": stage_loss,
                "p_corr": self.error_metrics
                #"ACC": self.error_metrics.summarize(),
            }

         # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["p_corr"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=["p_corr"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
    

    def init_optimizers(self):
        self.vit_optimizer = self.hparams.vit_opt_class(
            self.modules.vit_model.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "vit_opt", self.vit_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "vit_optimizer": self.vit_optimizer,
        }

    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def calc_pearsons(self, preds, labels):
        if not type(preds) == np.ndarray:
            print(type(preds))
            print(len(preds))
            preds = np.concatenate(preds)
        if not type(labels) == np.ndarray:
            labels = np.concatenate(labels)
        r = pearsonr(preds, labels)
        return r[0]


    def run_inference(
            self,
            dataset, # Must be obtained from the dataio_function
            max_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(max_key=max_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)


        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                labels = batch.label_required
                labels = torch.tensor(labels).to(self.device)

                decoder_outputs, wav2vec2_outputs, vit_feat = predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                decoder_outputs = decoder_outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                all_predictions.append(decoder_outputs)
                all_labels.append(labels)

            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)
            mean_r_feat = np.array([pearsonr(all_predictions[:, i], all_labels[:, i])[0] for i in range(all_predictions.shape[1])])
            mean_r = np.mean(mean_r_feat)

            print("Pearson correlation per feature: ", mean_r_feat)
            print("Pearson correlation: ", mean_r)
               

def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel.json"), replacements={"data_root": data_folder})
    
    datasets = [train_data, valid_data, test_data]

    # 1. Define audio pipeline:
    @sb.utils.data_pipeline.takes("audio_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(audio_path):
        sig, sr = librosa.load(audio_path, sr=16000)

        #take the first/last 10 seconds of the audio
        # sig = sig[-16000*10:]
        #sig = sig[:16000*10]
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 2. Define video pipeline:
    @sb.utils.data_pipeline.takes("audio_path")
    @sb.utils.data_pipeline.provides("vit_feat")
    def video_pipeline(audio_path):
        video_path = audio_path.replace("raw/wav", "feature_segments/vit-fer/")
        video_path = video_path.replace(".wav", ".csv")
        vit_feat = []
        with open(video_path, "r") as f:
            lines = f.readlines()
            feats = lines[1]
        
        for elem in feats.split(","):
            vit_feat.append(float(elem))

        # Convert to tensor
        vit_feat = torch.tensor(vit_feat)
        vit_feat = vit_feat[2:]
        return vit_feat

    # sb.dataio.dataset.add_dynamic_item(datasets, video_pipeline)

    # 3. Define image pipeline:
    @sb.utils.data_pipeline.takes("audio_path")
    @sb.utils.data_pipeline.provides("images")
    def image_pipeline(audio_path):
        image_path = audio_path.replace("raw/wav", "raw/faces")
        image_path = image_path.replace(".wav", "")
        images = []
        for elem in os.listdir(image_path):
            if elem.endswith(".jpg"):
                with PIL.Image.open(os.path.join(image_path, elem)) as image:
                    images.append(image.convert("RGB"))

        return images

    sb.dataio.dataset.add_dynamic_item(datasets, image_pipeline)


    # 4. Define text pipeline:
    @sb.utils.data_pipeline.takes("label_required")
    @sb.utils.data_pipeline.provides("label_required")
    def text_pipeline(label_required):
        yield label_required

    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "images", "label_required"])
    
    train_data = train_data.filtered_sorted(sort_key="duration", reverse=False)
    
    return train_data, valid_data, test_data


def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Freeze the parameters of ViT model
    for param in hparams["modules"]["vit_model"].parameters():
        param.requires_grad = False
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Trainer initialization
    ser_brain = SER(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

    # Dataset creation
    train_data, valid_data, test_data = data_prep("../data/audio", hparams)

    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###ser_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        ser_brain.fit(
            ser_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # evaluate
        print("Evaluating...")
        # calculate the time in takes to do the evaluation
        import time
        start_time = time.time()
        ser_brain.run_inference(test_data, "p_corr", hparams["test_dataloader_options"])
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
