import pandas as pd
import warnings
import argparse
import json
import os
import sys
from sklearn.model_selection import train_test_split

from data_utils import (
    load_data,
    preprocess_data,
    prepare_binary_labels,
    prepare_multiclass_labels,
    scale_features,
)
from classifiers import Classifier
from plots import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_class_distribution,
)

warnings.filterwarnings("ignore")


def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_results(model_name, classification_type, results):
    print(f"\n{model_name} ({classification_type}):")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results["report"])


def run_binary_classification(X_train, X_test, y_train, y_test, model_types, model_params):
    print("\n[BINARY CLASSIFICATION]")
    results = []

    for model_type in model_types:
        params = model_params.get(model_type, {})
        classifier = Classifier(model_type=model_type, num_classes=2, model_params=params)
        classifier.train(X_train, y_train)
        eval_results = classifier.evaluate(X_test, y_test, target_names=["BENIGN", "ATTACK"])
        print_results(classifier.name, "Binary Classification", eval_results)

        results.append(
            {
                "model_name": classifier.name,
                "model_type": model_type,
                "classification_type": "Binary",
                "accuracy": eval_results["accuracy"],
                "f1_score": eval_results["f1_score"],
                "predictions": eval_results["predictions"],
            }
        )

    return results


def run_multiclass_classification(
    X_train, X_test, y_train, y_test, class_names, model_types, model_params
):
    print("\n[MULTICLASS CLASSIFICATION]")
    results = []
    models_importance = []

    for model_type in model_types:
        params = model_params.get(model_type, {})
        classifier = Classifier(model_type=model_type, num_classes=len(class_names), model_params=params)
        classifier.train(X_train, y_train)
        eval_results = classifier.evaluate(X_test, y_test, target_names=class_names)
        print_results(classifier.name, "Multiclass Classification", eval_results)

        results.append(
            {
                "model_name": classifier.name,
                "model_type": model_type,
                "classification_type": "Multiclass",
                "accuracy": eval_results["accuracy"],
                "f1_score": eval_results["f1_score"],
                "predictions": eval_results["predictions"],
            }
        )

        importances = classifier.get_feature_importances()
        models_importance.append({"name": classifier.name, "importances": importances})

    return results, models_importance


def save_results(binary_results, multiclass_results, output_path):
    all_results = binary_results + multiclass_results

    results_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Classification Type": r["classification_type"],
                "Accuracy": r["accuracy"],
                "F1-score": r["f1_score"],
            }
            for r in all_results
        ]
    )

    print_header("RESULTS SUMMARY")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path.split('/')[-1]}")


def check_prerequisites(file_paths):
    print("\n[0/7] Checking prerequisites...")
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"✗ Error: Data file not found at {file_path}", file=sys.stderr)
            sys.exit(1)
    print("✓ All files and directories are in place.")


def main():
    parser = argparse.ArgumentParser(description="DDoS Attack Classification")
    parser.add_argument('--smaller-dataset', '-s', action='store_true', help='Use smaller dataset for quick run')
    args = parser.parse_args()

    print_header("DDOS ATTACK CLASSIFICATION")

    with open('config.json', 'r') as f:
        model_params = json.load(f)

    file_paths = [
        "data/Syn.csv",
        "data/NetBIOS.csv",
        "data/MSSQL.csv",
        "data/LDAP.csv",
        "data/UDPLag.csv",
        "data/UDP.csv",
        "data/Portmap.csv",
    ]

    if args.smaller_dataset:
        print("\nRunning with smaller dataset...")
        file_paths = [f.replace('.csv', '_small.csv') for f in file_paths]

    check_prerequisites(file_paths)

    model_types = ["rf", "xgb", "svm"]

    print("\n[1/7] Loading data...")
    data = load_data(file_paths)

    print("\n[2/7] Preprocessing data...")
    data = preprocess_data(data)
    print(f"Features: {data.shape[1] - 1}")
    print("\nClass distribution:")
    for label, count in data["Label"].value_counts().items():
        print(f"  {label}: {count} ({count / len(data) * 100:.1f}%)")

    X = data.drop("Label", axis=1)
    y = data["Label"]

    print("\n[3/7] Preparing binary classification...")
    y_binary = prepare_binary_labels(y)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    X_train_bin_scaled, X_test_bin_scaled = scale_features(X_train_bin, X_test_bin)

    binary_results = run_binary_classification(
        X_train_bin_scaled, X_test_bin_scaled, y_train_bin, y_test_bin, model_types, model_params
    )

    print("\n[4/7] Preparing multiclass classification...")
    y_multi, label_encoder = prepare_multiclass_labels(y)
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.3, random_state=42, stratify=y_multi
    )
    X_train_multi_scaled, X_test_multi_scaled = scale_features(
        X_train_multi, X_test_multi
    )

    multiclass_results, models_importance = run_multiclass_classification(
        X_train_multi_scaled,
        X_test_multi_scaled,
        y_train_multi,
        y_test_multi,
        label_encoder.classes_,
        model_types,
        model_params,
    )

    print("\n[5/7] Generating visualizations...")

    viz_results = []
    for bin_res, multi_res in zip(binary_results, multiclass_results):
        viz_results.append(
            {
                "model_name": bin_res["model_name"],
                "y_test_bin": y_test_bin,
                "y_pred_bin": bin_res["predictions"],
                "y_test_multi": y_test_multi,
                "y_pred_multi": multi_res["predictions"],
                "class_names": label_encoder.classes_,
            }
        )

    plot_confusion_matrices(viz_results, "outputs/confusion_matrices.png")
    plot_feature_importance(
        models_importance, X.columns, "outputs/feature_importance.png"
    )
    plot_class_distribution(y, "outputs/class_distribution.png")

    print("\n[6/7] Saving results...")
    save_results(
        binary_results,
        multiclass_results,
        "outputs/classification_results.csv",
    )

    print("\n[7/7] Analysis completed successfully!")
    print_header("ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
