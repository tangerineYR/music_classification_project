from Source import Classification
from Source import CreateDataset
from Source import TrainModel
from Source import EvaluateModel


def main():
    CreateDataset.main()
    TrainModel.main();
    Classification.main();

if __name__ == '__main__':
    main()
