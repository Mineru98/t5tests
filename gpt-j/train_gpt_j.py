import sys, os, argparse, transformers, torch, random
from datasets import load_dataset, load_metric, load_from_disk, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.contrib.concurrent import process_map
from transformers import  GPTJForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.optimization import AdafactorSchedule
from transformers.trainer_pt_utils import get_parameter_names, nested_detach
from bitsandbytes.optim import Adam8bit
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from gpt_j_8bit import GPTJForCausalLM8, GPTJBlock8, add_adapters

"""
# fp16 model, 8bit loading
    State must contain either CBt or CB matrix for backward

# fp16 model, not load_in_8bit 
    working.
    
# 8bit model, torch.float16 loading
    workging?

# 8bit model, no parameter loading= torch.float32 loading
    oom

"""

original_model = False
if original_model:
    tokenizer_name = "tokenizer-gpt-j-6B-org"
    base_model_name = "gpt-j-6B-8bit-org"
else:
    tokenizer_name = "tokenizer-gpt-j-plus-ko"
    base_model_name = "gpt-j-6B-ko-voc-to-8bit-conv"

num_train_epochs = 2
dataset_source = "wiki"
max_input_length = 128
continue_train = False
training_size = 0  # 0 means all
batch_size = 8    # 0 means auto
validation_data_size = batch_size * 10
# base_model_type = "int8"    # fp16, int8
base_model_type = "fp16"    # fp16, int8
load_in_8bit = False

model_name = None 
model_save_dir = None
train_dataloader = None 
eval_dataloader = None
# accelerate
accelerator = Accelerator(
    DistributedDataParallelKwargs(find_unused_parameters=True), 
    # mixed_precision="fp16"
)
device = accelerator.device

# tokenizer
tokenizer = None

class TextDataset(Dataset):
    def tokenizing_sample(self, s):
        tt = tokenizer(s[self.feature_name], max_length=max_input_length, truncation=True, padding=True)
        return tt
            
    def __init__(self, tokenize: bool = False):
        accelerator.print("reading dataset...", dataset_source)
        if dataset_source == "sns":
            ds = load_dataset("json", data_files="/home/chang/nas1/linux/dataset/text/한국어 SNS/korean_sns_training_gpt2_v2.json")
            feature_name = "sample"
        elif dataset_source == "wiki":
            wiki_local = "/home/chang/nas1/linux/dataset/text/wikipedia/20221001.kr"
            if os.path.exists(wiki_local):
                ds = load_from_disk(wiki_local)
            else:
                ds = load_dataset("lcw99/wikipedia-korean-20221001")
            feature_name = "text"
        elif dataset_source == "cc100":
            ds = load_dataset("cc100", lang="ko")
            feature_name = "text"
        elif dataset_source == "namu":
            ds = load_dataset("heegyu/namuwiki-extracted")
            feature_name = "text"
        self.feature_name = feature_name
        
        ds = ds["train"]
        if training_size > 0:
            ds = ds.select(range(training_size))
        accelerator.print("reading dataset done...", dataset_source)

        if tokenize:
            ds = ds.map(self.tokenizing_sample, batched=True)
            examples = ds['input_ids']
            
            # examples = []
            # num_worker = 10
            # chunk_size = int(len(ds) / (num_worker * 4))
            # if chunk_size > 100000:
            #     chunk_size = 100000
            # accelerator.print("chunk_size=", chunk_size)
            # for result in process_map(self.tokenizing_sample, ds, max_workers=num_worker, chunksize=chunk_size):
            #     examples += result
        else:
            examples = ds[self.feature_name]
        self.data_list = examples

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def get_dataloaders(tokenize: bool = False, loader_batch_size: int = batch_size):
    dataset = TextDataset(tokenize=tokenize)
    train_dataset = dataset[validation_data_size:]
    eval_dataset = dataset[:validation_data_size]
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=lambda x: x)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=loader_batch_size, collate_fn=lambda x: x)
    return train_dataloader, eval_dataloader

def build_tokenizer():
    tokenizer_path = f"../train_tokenizer/{tokenizer_name}"
    accelerator.print("\n-----------------------\ntokenizer path = ", tokenizer_path)
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        accelerator.print("downloading tokenizer from hf")
        tokenizer = AutoTokenizer.from_pretrained("lcw99/tokenizer-gpt-j-ext-ko")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_adam8bit_optimizer(model):
    training_args = TrainingArguments(per_device_train_batch_size=batch_size, output_dir=".")

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim
        
def init_model():
    if base_model_type == "fp16":
        max_memory_mapping = {0: "10GB", 1: "10GB"}
        gpt = GPTJForCausalLM.from_pretrained(
            "./StockModels/gpt-j-6B-fp16-ko-voc",
            #revision="float16",
            torch_dtype=torch.float16,
            device_map='auto',
            load_in_8bit=load_in_8bit,
            #max_memory=max_memory_mapping,
            #low_cpu_mem_usage=True,
            use_cache=False,
        )
        # tokenizer_len = len(tokenizer)
        # print("\n\n\n=====\ntokenizer_len=", tokenizer_len)
        # gpt.resize_token_embeddings(tokenizer_len)
        # print("resize done....")
        gpt.gradient_checkpointing_enable()
        
        gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit"
        gpt.config.__dict__["use_cache"] = False
        # gpt.save_pretrained("./StockModels/gpt-j-6B-8bit-ko-voc")
        if True:
            optimizer = Adam8bit(gpt.parameters(), lr=5e-5)
            #optimizer = build_adam8bit_optimizer(gpt)
        else:
            optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)
    else:
        # transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock8  
        model_path = f"./StockModels/{base_model_name}"
        if os.path.exists(model_path):
            accelerator.print("base model path = ", model_path)
            gpt =  GPTJForCausalLM.from_pretrained(
                model_path,
                # torch_dtype=torch.float16,
            )
        else:
            hf_model = "lcw99/gpt-j-6B-voc-ext-to-91238-8bit"
            accelerator.print("downloading..", hf_model)
            gpt = GPTJForCausalLM.from_pretrained(hf_model)
        # add_adapters(gpt)
        gpt.gradient_checkpointing_enable()
        gpt.config.__dict__["_name_or_path"] = "lcw99/gpt-j-6B-8bit-custom"

        optimizer = Adam8bit(gpt.parameters(), lr=1e-5, weight_decay=0.01)
        
    gpt.config.__dict__["use_cache"] = False

    return gpt, optimizer
              
def _get_lr(param_group, param_state):
    step = param_state["step"]
    eps = param_group["eps"]
    return 5e-5 - eps * step * 1e-2
              
def loss_function(output, input):
    # loss_cross_entropy = nn.CrossEntropyLoss()
    # loss = loss_cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten()) 
    # return loss
    # else:
    # if output.loss:
    #     return output.loss
    loss = F.cross_entropy(output.logits[:, :-1, :].flatten(0, -2), input['input_ids'][:, 1:].flatten(), reduction='mean')
    return loss
    
def trainer():
    global batch_size
    if batch_size == 0:
        batch_size = 4
    model, optimizer = init_model()
    train_dataloader, eval_dataloader = get_dataloaders(tokenize=False, loader_batch_size=batch_size)
 
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(num_training_steps*0.1), num_training_steps
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        batch_token = tokenizer(batch, truncation=True, padding=True, max_length=max_input_length, return_tensors='pt')
        with torch.cuda.amp.autocast():
            outputs = model(**batch_token.to(device))
            loss = loss_function(outputs, batch_token)
            # loss = loss.mean()  # mean() to average on multi-gpu parallel training
        accelerator.backward(loss)
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # scaler.step(optimizer)
        # scaler.update()
                
        lr_scheduler.step()
        if i % 10 == 0 and i != 0:
            state = {
                'k' : i, 'epoch': num_epochs, 
                'lr_scheduler': lr_scheduler.state_dict(), 
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }   
            print(loss)
            # torch.save(state, "./TestModel/model.pt")

class MyTrainer(Trainer):    
    # def create_optimizer_and_scheduler(self, num_training_steps):
    #     self.optimizer = Adam8bit(self.model.parameters(), lr=1e-5)
    #     self.lr_scheduler = AdafactorSchedule(self.optimizer)    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        #inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt').to(device)

        #outputs = model.forward(**inputs)
        
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            loss = self.scaler.scale(loss)
            accelerator.backward(loss)
            # self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            #loss.backward()
            accelerator.backward(loss)
        # loss.backward(retain_graph=True)

        return torch.squeeze(loss.detach())

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = None

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, [])

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    def compute_loss(self, model, inputs, return_outputs=False):
        # return super(MyTrainer, self).compute_loss(outputs, inputs, return_outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        outputs = model(**inputs)
        if "loss" in outputs:
            loss = outputs["loss"]
        else:
            loss_cross_entropy = nn.CrossEntropyLoss()
            loss = loss_cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten()) 
            # loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), inputs['input_ids'][:, 1:].flatten(),
            #                    reduction='mean')
        #print("loss=", loss)
        return (loss, outputs) if return_outputs else loss
    
    # def get_train_dataloader(self) -> DataLoader:
    #     return accelerator.prepare(train_dataloader)

    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    #     return accelerator.prepare(eval_dataloader)
        
def compute_metrics(eval_pred):
    # preds, labels = eval_pred
    # # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # # by preprocess_logits_for_metrics
    # labels = labels.reshape(-1)
    # preds = preds[0].reshape(-1)
    # mask = labels != -100
    # labels = labels[mask]
    # preds = preds[mask]
    # acc = metric_accuracy.compute(predictions=preds, references=labels)
    # return acc
        
    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    # labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    # label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    ppl = {}
    ppl["mean_perplexity"] = 0.0

    #ppl = perplexity.compute(predictions=pred_str_filterd, model_id='gpt2')

    print("\n===========predictions\n", pred_str)
    # print()
    # for label in random.sample(list(label_str), 2):
    #     print("label=", label)

    return {
        "mean_perplexity": round(ppl["mean_perplexity"], 4)
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    l0 = logits[0]
    pred_ids = torch.argmax(l0, dim=-1)
    #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    return pred_ids, labels

def huggingface_trainer():
    global batch_size, train_dataloader, eval_dataloader

    model, optimizer = init_model()
    train_dataloader, eval_dataloader = get_dataloaders(tokenize=True, loader_batch_size=batch_size)
 
    # num_training_steps = num_train_epochs * len(train_dataloader)
    # lr_scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer, 500, num_training_steps
    # )

    lr_scheduler = AdafactorSchedule(optimizer)    
    optimizer._get_lr = _get_lr

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    
    if batch_size == 0:
        auto_find_batch_size = True
        batch_size = 2
    else:
        auto_find_batch_size = False
    
    args = TrainingArguments(
        model_save_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        # learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=auto_find_batch_size,
        gradient_accumulation_steps=1,
        weight_decay=0.02,
        save_total_limit=5,
        num_train_epochs=num_train_epochs,
        #predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        ignore_data_skip=True,     # set true for ignore batch skip, fast
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, return_tensors="pt", mlm=False)
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset = train_dataloader.dataset,
        eval_dataset = eval_dataloader.dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer = accelerator.prepare(trainer)
    print("start trainning -----------------------------")
    if continue_train:
        trainer.train(True)
    else:
        trainer.train()
    trainer.save_model()
    
                                    
def main():
    global model_save_dir, dataset_source, tokenizer_name, max_input_length, continue_train, training_size, batch_size, tokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--continue_train", action='store_true', help = "continue trainning")
    parser.add_argument("-d", "--dataset", help = "dataset source = [sns, wiki, cc100, namu]")
    parser.add_argument("-t", "--tokenizer", help = "tokenizer name")
    parser.add_argument("-i", "--max_input_length", help = "max input length")
    parser.add_argument("-s", "--training_size", help = "training size, 0 for all")
    parser.add_argument("-b", "--batch_size", help = "batch size, 0 for auto")

    args = parser.parse_args()

    if args.dataset:
        dataset_source = args.dataset
    else:
        parser.print_help(sys.stderr)
        return

    if args.tokenizer:
        tokenizer_name = args.tokenizer

    if args.continue_train:
        print("=== param continue trainning")
        continue_train = True

    if args.max_input_length:
        max_input_length = int(args.max_input_length)
    if args.training_size:
        training_size = int(args.training_size)
    if args.batch_size:
        batch_size = int(args.batch_size)
     
    model_name = f'{base_model_name}_{tokenizer_name}_{dataset_source}' 
    model_save_dir = f"./Models/{model_name}"
        
    tokenizer = build_tokenizer()
    
    # trainer()
    huggingface_trainer()
    
if __name__ == '__main__':
    sys.exit(main()) 