import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphtask import Task
from pyod.models.lof import LOF
from timeit import Timer


def run_experiment(
        n_jobs: int = 1,
        n_datasets=1,
        n_features=128,
        n_samples=1024,
        n_bagging=1
):
    max_features = 0.5
    max_samples = 0.5
    task = Task(n_jobs=n_jobs)

    dataset_names = []
    for dataset in range(n_datasets):
        dataset_name = f"dataset_{dataset + 1}"
        task.step(lambda: np.random.randn(n_samples, n_features), rename=dataset_name)
        dataset_names.append(dataset_name)

    @task.step(args=dataset_names)
    def gather_data(*args):
        return list(zip(dataset_names, args))

    @task.step(map="gather_data", n_jobs=n_jobs, backend="threading")
    def feature_bagging(gather_data):
        assert 0 < max_features <= 1
        name, data = gather_data
        _, n_features = data.shape
        n_bagging_features = int(n_features * max_features)
        assert n_bagging_features > 0
        for _ in range(n_bagging):
            yield np.random.choice(n_features, n_bagging_features)

    @task.step(map="gather_data", n_jobs=n_jobs, backend="threading")
    def samples_bagging(gather_data):
        assert 0 < max_samples <= 1
        name, data = gather_data
        n_samples, _ = data.shape
        n_bagging_samples = int(n_samples * max_samples)
        assert n_bagging_samples > 0
        for _ in range(n_bagging):
            yield np.random.choice(n_features, n_bagging_samples)

    @task.step
    def gather_bagging(gather_data, feature_bagging, samples_bagging):
        for ((name, data), feature_idx, samples_idx) in zip(gather_data, feature_bagging, samples_bagging):
            for fi, si in zip(feature_idx, samples_idx):
                yield name, data[si, :][:, fi]

    @task.step(map="gather_bagging", n_jobs=n_jobs, backend="loky")
    def predict_score(gather_bagging):
        name, data = gather_bagging
        score = LOF(n_jobs=1).fit(data).decision_scores_
        return name, score

    def run_parallel():
        start = time.time()
        task()
        end = time.time()
        return end - start

    def run_sequential():
        start = time.time()
        datasets = [np.random.randn(n_samples, n_features) for _ in range(n_datasets)]
        gathered_data = gather_data(*datasets)
        feature_idx = feature_bagging(gathered_data)
        samples_idx = samples_bagging(gathered_data)
        gathered_idx = gather_bagging(gathered_data, feature_idx, samples_idx)
        predict_score(gathered_idx)
        end = time.time()
        return end - start

    return print(run_parallel() if n_jobs > 1 else run_sequential())


def time_experiment(n_jobs):
    fn = lambda: run_experiment(
        n_jobs=n_jobs,
        n_datasets=10,
        n_bagging=10
    )
    timer = Timer(fn)
    # the first measurement creates the process pool as a warm-up
    times = timer.repeat(repeat=25, number=1)
    return np.mean(times), np.std(times)


def plot_practical(df):
    fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
    n_jobs, mt, st = list(range(1, 13)), df["mean"].values, df["std"].values
    ax.plot(mt, label="time (mean)", c="black")
    ax.fill_between(
        range(len(df)),
        mt - st,
        mt + st,
        color="lightgray",
        label="time (std)"
    )

    ax.set_ylabel("seconds")
    ax.set_xlabel("workers")
    ax.set_xticks(range(len(df)), labels=n_jobs)
    ax.legend()
    return fig


if __name__ == "__main__":
    result = []
    for n in range(1, 13):
        mean, std = time_experiment(n_jobs=n)
        result.append({"mean": mean, "std": std})

    df = pd.DataFrame(result)
    df.to_csv("practical-experiment.csv")

    # to reproduce the plots in the paper, run:
    # plot_practical(df)
