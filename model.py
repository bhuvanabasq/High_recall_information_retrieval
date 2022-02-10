from simpletransformers.classification import ClassificationModel, ClassificationArgs
from rq import get_current_job


def create_model_and_train(train_df, username, reset_train):
    class_weights = (1 / train_df['labels'].value_counts(normalize=True)).to_list()
    model_args = ClassificationArgs(num_train_epochs=1, output_dir="output/" + username, overwrite_output_dir=True)

    if reset_train:
        model = ClassificationModel("distilbert", "distilbert-base-uncased-finetuned-sst-2-english", args=model_args,
                                    weight=class_weights, use_cuda=False)
    else:
        model = ClassificationModel(
            "distilbert", "output/" + username, use_cuda=False, args=model_args,
            weight=class_weights)
    # train the model
    model.train_model(train_df)
    return "model trained"


def test_model(test_df, username):
    job = get_current_job()

    n = len(test_df) // 20  # chunk row size
    list_df = [test_df[i:i + n] for i in range(0, test_df.shape[0], n)]
    predictions = []

    model = ClassificationModel(
        "distilbert", "output/" + username, use_cuda=False
    )
    for i in range(len(list_df)):
        job.meta['progress'] = i / len(list_df) * 100
        job.save_meta()

        list_df_chunk = list_df[i]['text'].to_list()
        prediction_val, _ = model.predict(list_df_chunk)
        predictions.append(prediction_val)

    predictions_flat = [item for sublist in predictions for item in sublist]

    test_df["prediction"] = predictions_flat
    test_df.to_csv('predict_' + username)
    return 0
