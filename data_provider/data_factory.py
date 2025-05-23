from data_provider.data_loader import Dataset_Custom
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader



def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last =False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq

    if args.task_name == 'classification':
        drop_last = True
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    return
