import torch
import json
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_inference_time(model, Input):
    model.eval()
    with torch.no_grad():
        start = time.time()
        output = model(Input)
        inference_time = time.time() - start
    return round(inference_time, 6)


def main():
    f = open('models\\name.txt', 'r')
    js = f.read()
    model_li = json.loads(js)
    print(model_li)
    f.close()

    batch_size_li = [1, 2, 4, 8, 16, 32, 64]

    infer_time_dic = {}

    for name in model_li:
        print("=> Compute model {}'s inference time".format(name))
        model = load_model(name)
        single_model_time = {}
        for batch in batch_size_li:
            Input = torch.rand([batch, 3, 224, 224]).to(device)
            print("Random imput size is {}".format([batch, 3, 224, 224]))
            time = compute_inference_time(model, Input)
            torch.cuda.empty_cache()
            single_model_time[batch] = time
        infer_time_dic[name] = single_model_time
        print(infer_time_dic)
    with open('result/inference_time.txt', 'w') as f:
        js = json.dumps(infer_time_dic)
        f.write(js)
    f.close()
    print('done!')

def load_model(name):
    path = 'models\\' + name + '.pth'
    model = torch.load(path)
    model.to(device)
    return model


if __name__ == '__main__':
    main()