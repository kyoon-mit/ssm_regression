from pathlib import Path
import yaml
import torch
from model import LitS4Model
from data import LitBNSDataModule

def maketensor(d):
    return {k: torch.tensor(v) for k, v in d.items()}

def makesqrttensor(d):
    return {k: torch.sqrt(torch.tensor(v)) for k, v in d.items()}

def dinit(prefix, variables):
    d = {f'{prefix}_{var}': [] for var in variables}
    return d
    
def dump_to_csv(outputs, csv_name):
    import pandas as pd
    df = pd.DataFrame(outputs)
    df.to_csv(csv_name, index=False)
    print(f'Dumped results to {csv_name}')
    return df

def load_config(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    data_init_args = cfg.get('data', {}).get('init_args', {})
    model_init_args = cfg.get('model', {}).get('init_args', {})
    variables = data_init_args.get('variables')
    if not isinstance(variables, list):
        variables = list(variables)
        data_init_args['variables'] = variables
    print(f'Variables: {variables}.')
    return data_init_args, model_init_args, variables

def compute_vals(checkpoint_path, config_path, csv_name, compute_on_cpu=False):
    data_init_args, model_init_args, variables = load_config(config_path)

    model = LitS4Model(**model_init_args)
    checkpoint = torch.load(checkpoint_path, weights_only=True,
                            map_location=torch.device('cpu' if compute_on_cpu else 'cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    bns_data_module = LitBNSDataModule(**data_init_args)
    bns_data_module.setup('test')
    test_data_loader = bns_data_module.test_dataloader()

    # Placeholders for values
    pred_dict, truth_dict, pred_sigma_dict, return_dict =\
        dinit('pred', variables), dinit('truth', variables), dinit('sigma', variables), dict()

    for batch in test_data_loader:
        h1, l1, params, idx = batch
        device = h1.device
        inputs = torch.stack([h1.to(device), l1.to(device)], dim=2)
        truths = torch.stack(list(params.values()), dim=1)

        with torch.no_grad():
            preds = model(inputs)

        # Move samples to CPU if required
        if compute_on_cpu:
            preds = preds.cpu()
            truths = truths.cpu()
        
        for i in range(len(variables)):
            p = variables[i]
            pred_dict[f'pred_{p}'].extend(preds[:, i].tolist())
            truth_dict[f'truth_{p}'].extend(truths[:, i].tolist())
            # TODO: implement for other losses
            pred_sigma_dict[f'sigma_{p}'].extend(preds[:, i+len(variables)].tolist())

        return_dict.update(maketensor(pred_dict))
        return_dict.update(maketensor(truth_dict))
        # TODO: implement for other losses
        return_dict.update(makesqrttensor(pred_sigma_dict))
        dump_to_csv(return_dict, csv_name=csv_name)

        return return_dict
    
def main():
    base_path = Path('/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning')
    checkpoint_path = base_path / 'lightning_logs/c3p6di6k/checkpoints/ckpt_epoch=18-step=0969.ckpt'
    config_path = base_path / 'config.yaml'
    csv_path = base_path / 'bns_outputs.csv'
    results = compute_vals(checkpoint_path, config_path, csv_path, compute_on_cpu='True')
    return

if __name__ == '__main__':
    main()

