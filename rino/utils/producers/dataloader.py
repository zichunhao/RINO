from pathlib import Path
import multiprocessing
from dataloader import jetclass, processed
from ..logger import LOGGER
from accelerate import Accelerator

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_dataloader_and_config(
    config: dict,
    split: str,
    mode: str,  # training or inference
    accelerator: Accelerator | None = None,
) -> (
    tuple[jetclass.JetDataLoader, jetclass.DataloaderConfig]
    | tuple[processed.JetDataLoader, processed.DataloaderConfig]
):
    n_threads = multiprocessing.cpu_count()

    if mode not in config:
        raise ValueError(f"Mode {mode} not found in config for dataloader.")
    shuffle_dict = config.get("shuffle", None)
    if shuffle_dict is None:
        shuffle = True
    else:
        shuffle = shuffle_dict.get(mode, True)
    LOGGER.info(f"Dataloader (split={split}) shuffle: {shuffle}")

    try:
        config_dataloader = config[mode]["dataloader"][split]
    except KeyError:
        config_dataloader = config[mode]["dataloader"]["default"]
    
    # Get num_workers from config
    num_workers = config_dataloader.get("num_workers", None)
    if num_workers is None:
        num_workers = n_threads

    # DISPATCH BATCHES LOGIC: Adjust num_workers based on accelerator settings
    if accelerator is not None:
        # Check if using dispatch_batches (only main process needs workers)
        dispatch_batches = getattr(accelerator.dataloader_config, 'dispatch_batches', False)
        
        if dispatch_batches:
            if accelerator.is_main_process:
                LOGGER.info(f"Main process with dispatch_batches=True: using {num_workers} workers")
            else:
                # Worker processes: no workers needed (they don't load data)
                num_workers = 0
                LOGGER.info(f"Worker process {accelerator.process_index} with dispatch_batches=True: using 0 workers")
        else:
            # Standard distributed training: reduce workers per process
            num_workers = max(1, num_workers // accelerator.num_processes)
            LOGGER.info(f"Standard distributed training: reduced to {num_workers} workers per process")
    
    # For cached datasets, reduce workers further (data is in memory)
    cached = config_dataloader.get("cached", False)
    if cached and num_workers > 4:
        num_workers = min(num_workers, 4)
        LOGGER.info(f"Cached dataset: reduced workers to {num_workers} (data is in memory)")

    dataloader_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    LOGGER.info(f"Dataloader (split={split}) number of workers: {num_workers}")
    
    prefetch_factor = config_dataloader.get("prefetch_factor", None)
    if prefetch_factor is not None:
        if num_workers > 0:
            # Validate reasonable range
            if prefetch_factor < 1:
                LOGGER.warning(f"Prefetch factor {prefetch_factor} < 1, using default (2)")
                prefetch_factor = 2
            elif prefetch_factor > 10:  # or whatever max makes sense for your use case
                LOGGER.warning(
                    f"Prefetch factor {prefetch_factor} is quite high; may use excessive memory"
                )

            dataloader_kwargs["prefetch_factor"] = prefetch_factor
            LOGGER.info(f"Dataloader (split={split}) prefetch factor: {prefetch_factor}")
        else:
            LOGGER.warning(
                f"Prefetch factor {prefetch_factor} is set, but num_workers is 0. "
                "Ignoring prefetch factor."
            )

    kwargs = config_dataloader.get("kwargs", None)
    if kwargs is None:
        kwargs = {}
        LOGGER.debug(f"No dataloader kwargs found in training config. ")
    else:
        LOGGER.debug(f"Dataloader kwargs from training config: {kwargs}")

    preprocessed = config_dataloader.get("preprocessed", False)
    dataloader_config = get_config(
        config, mode=mode.lower(), split=split, preprocessed=preprocessed
    )
    
    if "train" in split:
        data_split = "train"
    elif "val" in split:
        data_split = "val"
    elif "test" in split:
        data_split = "test"
    else:
        raise ValueError(f"Unrecognized split: {split}. Split must contain 'train', 'val', or 'test'.")

    if preprocessed:
        if cached:
            dataset = processed.CachedJetDataset(
                config=dataloader_config, split=data_split, accelerator=accelerator, **kwargs
            )
        else:
            dataset = processed.JetDataset(
                config=dataloader_config, split=data_split, **kwargs
            )
    else:
        if cached:
            dataset = jetclass.CachedJetDataset(
                config=dataloader_config, split=data_split, accelerator=accelerator, **kwargs
            )
        else:
            dataset = jetclass.JetDataset(config=dataloader_config, split=data_split, **kwargs)

    LOGGER.info(
        f"Dataloader (preprocessed={preprocessed}, cached={cached}, split={split}) config: {config_dataloader}"
    )

    if preprocessed:
        _bsz = 1  # handled internally
        dataloader = processed.JetDataLoader(
            dataset,
            shuffle=shuffle,
            # batch_size=_bsz,
            pin_memory=True,
            **dataloader_kwargs,
        )
    else:
        _bsz = dataset.config.batch_size // dataset.config.batch_size_atomic
        dataloader = jetclass.JetDataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=_bsz,
            pin_memory=True,
            **dataloader_kwargs,
        )
    LOGGER.info(f"Dataloader (split={split}): {dataloader}")

    return dataloader, dataloader_config


def get_config(
    config: dict, 
    mode: str, 
    split: str | None = None,
    preprocessed: bool = False  # whether the config is for pre-processed data
) -> jetclass.DataloaderConfig | processed.DataloaderConfig:
    # Load dataloader config
    dataloader_config = config[mode]["dataloader"]
    if not split:
        split = iter(dataloader_config.keys()).__next__()
        LOGGER.debug(f"No split specified. Using first split: {split}")
    try:
        config_dataloader_path = config[mode]["dataloader"][split]["config"]
    except KeyError:
        LOGGER.debug(
            f"No dataloader config found for split {split}. Using default."
        )
        config_dataloader_path = config[mode]["dataloader"]["default"]["config"]
    config_dataloader_path = config_dataloader_path.replace(
        "PROJECT_ROOT", str(PROJECT_ROOT)
    )
    if not Path(config_dataloader_path).exists():
        raise FileNotFoundError(
            f"Dataloader config not found: {config_dataloader_path}"
        )
    if preprocessed:
        config = processed.DataloaderConfig(path=config_dataloader_path)
    else:
        config = jetclass.DataloaderConfig(path=config_dataloader_path)
    # LOGGER.debug(f"Dataloader config (split={split}): {config}")
    return config
