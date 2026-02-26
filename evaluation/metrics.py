def update_confusion_matrix(expected_type, response):

    reject_phrase = "only help with travel"

    if expected_type == "reject":
        if reject_phrase in response:
            return "TN"
        else:
            return "FP"

    else:
        if reject_phrase in response:
            return "FN"
        else:
            return "TP"
