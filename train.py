from model import Model

if __name__ == '__name__':
    model = Model()
    model.fit(saved_data = False)
    model.eval()