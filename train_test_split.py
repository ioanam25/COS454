import numpy as np
import pandas as pd
def train_test_split(images: np.ndarray,  details: pd.DataFrame):
    X_train = np.empty([2,2]) #shape??
    X_test = np.empty([2,2])
    y_train = np.empty([1])
    t_test = np.empty([1])
    for index, image in enumerate(images):
        if details.iloc[index]["color"] in ["green", "yellow"] and details.iloc[index]["shape"] in ["triangle", "star"]:
            X_test.append(image)
            y_test.append()??
        else:
            X_train.append(image)
            y_test.append()

    return X_train, y_train, X_test, y_test

# all images have color, shape, position (x, y pos of center), orientation(angle between origin and top middle point),
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


