# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# Other
import os
import glob

def find_last_checkpoint(callback_path, return_full_path=False):

    # All Checkpoints
    checkpoints = glob.glob(os.path.join(callback_path, "checkpoints_*.ckpt"))

    # Select Last Checkpoint else None
    max_steps = 0
    last_checkpoint = None
    for checkpoint in checkpoints:
        checkpoint = checkpoint.split("/")[-1]
        checkpoint_steps = int(checkpoint.split("_")[-1].replace(".ckpt", ""))
        if checkpoint_steps > max_steps:
            max_steps = checkpoint_steps
            last_checkpoint = checkpoint

    # Join path
    if last_checkpoint != None and return_full_path:
        last_checkpoint = os.path.join(callback_path, last_checkpoint)

    return last_checkpoint

def load_model(args, model, callback_path):

    # Model Device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.dreamerv3.twister.cpu else "cpu")
    if "cuda" in str(device):
        print("device: {}, {}, {}MB".format(device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
        args = args.update(_num_gpus=torch.cuda.device_count())
    else:
        print("device: {}".format(device))
        args = args.update(num_gpus=1)

    # Set Model Device
    model = model.to(device)

    # Append callback Tag
    if hasattr(args, "callback_tag"):
        callback_path = os.path.join(callback_path, args.callback_tag)

    # Last Checkpoint
    if hasattr(args, "load_last"):
        last_checkpoint = find_last_checkpoint(callback_path)
        if last_checkpoint != None:
            args = args.update(checkpoint=last_checkpoint)

    # Load Checkpoint
    if hasattr(args, "checkpoint"):
        model.load(os.path.join(callback_path, args.checkpoint))

    # Model Summary
    model.summary(show_dict=args.dreamerv3.twister.show_dict, show_modules=args.dreamerv3.twister.show_modules)
    
    return model

def load_datasets(training_dataset, evaluation_dataset):
    def print_dataset(dataset, tag):
        print("{} Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(tag, dataset.dataset.__class__.__name__, len(dataset.dataset), len(dataset), dataset.dataset.batch_size))

    # DataLoader
    dataset_train = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=training_dataset.batch_size,
        shuffle=training_dataset.shuffle,
        sampler=None,
        num_workers=training_dataset.num_workers,
        collate_fn=training_dataset.collate_fn,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=None,
        persistent_workers=training_dataset.persistent_workers,
    )
    
    # Loaded Print
    print_dataset(dataset_train, "Training")


        # Multiple Evaluation datasets
    if isinstance(evaluation_dataset, list):

        dataset_eval = []
        for dataset in evaluation_dataset:

            # DataLoader
            dataset_eval.append(torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=dataset.batch_size,
                shuffle=dataset.shuffle,
                sampler=None,
                num_workers=dataset.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=None,
                persistent_workers=dataset.persistent_workers,
            ))
        
            # Loaded Print
            print_dataset(dataset_eval[-1], "Evaluation")

    # One Evaluation dataset
    else:

        # DataLoader
        dataset_eval = torch.utils.data.DataLoader(
            dataset=evaluation_dataset,
            batch_size=evaluation_dataset.batch_size,
            shuffle=evaluation_dataset.shuffle,
            sampler=None,
            num_workers=evaluation_dataset.num_workers,
            collate_fn=evaluation_dataset.collate_fn,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=None,
            persistent_workers=evaluation_dataset.persistent_workers,
        )
        
        # Loaded Print
        print_dataset(dataset_eval, "Evaluation")
    
    return dataset_train, dataset_eval