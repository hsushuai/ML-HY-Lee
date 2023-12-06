def select_feat(train_data, valid_data, test_data, select_all=True):
    r"""Select useful features to perform regression
    :return (x_train, x_valid, x_test, y_train, y_valid)"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # TODO: Select suitable features columns
        feat_idx = list(range(35, raw_x_train.shape[1]))
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid
