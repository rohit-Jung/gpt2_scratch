from .loader import loadModelAndSaveWeights

def main() -> None :
    loadModelAndSaveWeights("../weights")
    print("Weights saved successfully!")
