import torch
import json

from thop import profile
from torch.backends import cudnn

from compute_accuracy import load_model

# define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    f = open('models\\name.txt', 'r')
    js = f.read()
    model_li = json.loads(js)
    print(model_li)
    f.close()

    complexity = {}

    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))

    for name in model_li:

        # create model
        model = load_model(name)

        dsize = (1, 3, 224, 224)
        if "inception" in name:
            dsize = (1, 3, 299, 299)
        inputs = torch.randn(dsize).to(device)
        total_ops, total_params = profile(model, (inputs,), verbose=False)
        data_dic = {}
        data_dic['Params(M)'] = total_params / (1000 ** 2)
        data_dic['FLOPS(G)'] = total_ops / (1000 ** 3)
        complexity[name] = data_dic
        print(
            "%s | %.2f | %.2f" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3))
        )

    with open('result/complexity.txt', 'w') as f:
        js = json.dumps(complexity)
        f.write(js)
        f.close()
    print('done!')

if __name__ == '__main__':
    main()
