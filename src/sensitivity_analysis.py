def compare_sensitivity(
    primary_results,
    sensitivity_results
):

    merged = primary_results.merge(
        sensitivity_results,
        on="model",
        suffixes=(
            "_primary",
            "_sensitivity"
        )
    )

    merged["auc_change"] = (
        merged["roc_auc_sensitivity"]
        -
        merged["roc_auc_primary"]
    )

    return merged