import os
import itertools
import ADLStream
from tqdm import tqdm

PERIODS = [10, 100, 250, 500, 750, 1000]  # [10, 100, 250, 500, 750, 1000]
CLOUDS = ["Clear sky", "Overcast", "Variable", "Very variable"]
SITES = ["Alderville"]  # , "Varennes"]

PAST_HISTORY = [300, 600, 1200]
FORECASTING_HORIZON = [60, 120, 300]

MODELS = {
    "mlp": {"hidden_layers": [256, 128],},
    "lstm": {
        "recurrent_units": [16, 32, 64],
        "recurrent_dropout": 0,
        "return_sequences": False,
        "dense_layers": [128, 64],
        "dense_dropout": 0,
    },
    "cnn": {
        "conv_layers": [16, 32, 64],
        "kernel_sizes": [3, 5, 7],
        "pool_sizes": [0, 0, 0],
        "dense_layers": [128, 64],
        "dense_dropout": 0.0,
    },
}

SHOW_PLOT = False
SAVE_PLOT = False
CHUNK_SIZE = 10
FADDING_FACTOR = 0.98
METRIC = "MAE"
MODEL_LOSS = "mae"
MODEL_OPTIMIZER = "adam"


def run_adls_experiment(
    site,
    clouds,
    period,
    past_history,
    forecasting_horizon,
    model_arch,
    model_params,
    model_id,
):
    datafilename = "./data/{}/{}_{}ms.csv".format(clouds, site, period)
    stream = ADLStream.data.stream.CSVFileStream(
        filename=datafilename, index_col=1, header=1, stream_period=period,
    )
    stream_generator = ADLStream.data.MovingWindowStreamGenerator(
        stream=stream,
        past_history=past_history,
        forecasting_horizon=forecasting_horizon,
    )

    results_path = "./results/ADLStream/{}/{}/{}ms/{}/{}/{}/".format(
        clouds, site, period, forecasting_horizon, past_history, model_arch
    )
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    evaluator = ADLStream.evaluation.PrequentialEvaluator(
        chunk_size=CHUNK_SIZE,
        metric=METRIC,
        fadding_factor=FADDING_FACTOR,
        results_file=results_path + "{}.csv".format(model_id),
        dataset_name="{} - {} ({}ms) - {} {}".format(
            site, clouds, period, model_arch, model_id
        ),
        show_plot=SHOW_PLOT,
        plot_file=results_path + "{}.jpg".format(model_id) if SAVE_PLOT else None,
    )

    adls = ADLStream.ADLStream(
        stream_generator=stream_generator,
        evaluator=evaluator,
        batch_size=90,
        num_batches_fed=60,
        model_architecture=model_arch,
        model_loss=MODEL_LOSS,
        model_optimizer=MODEL_OPTIMIZER,
        model_parameters=model_params,
        log_file="ADLStream.log",
    )

    adls.run()


site = SITES[0]
clouds = CLOUDS[0]
period = PERIODS[3]
past_history = PAST_HISTORY[0]
forecasting_horizon = FORECASTING_HORIZON[0]

for site, clouds, period, past_history, forecasting_horizon in tqdm(
    itertools.product(SITES, CLOUDS, PERIODS, PAST_HISTORY, FORECASTING_HORIZON)
):
    for model in MODELS:
        print("\n{}\n".format(model))
        run_adls_experiment(
            site=site,
            clouds=clouds,
            period=period,
            past_history=past_history,
            forecasting_horizon=forecasting_horizon,
            model_arch=model,
            model_params=MODELS[model],
            model_id=1,
        )
