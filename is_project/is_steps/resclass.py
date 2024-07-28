class Result:
    def __init__(self) -> None:
        """Initialize the Result object and reset its attributes."""
        self.reset()

    def reset(self) -> None:
        """Reset all attributes to their default values."""
        self.ids = 0
        self.predictions = 0
        self.dur_motion = {}
        self.dur_phone = {}
        self.max_value = 25
        self.motion_alert = {}
        self.phone_alert = {}
        self.histeresis_left = 10
        self.histeresis_right = 16

    def set(self, new_ids: int, new_predictions: int) -> None:
        """Set the ids and predictions attributes to new values.

        Args:
            new_ids (int): New value for ids.
            new_predictions (int): New value for predictions.
        """
        self.ids = new_ids
        self.predictions = new_predictions

    def set_duration(self, new_motion_dur: dict, new_phone_dur: dict) -> None:
        """Update the duration dictionaries for motion and phone usage.

        Args:
            new_motion_dur (dict): New durations for motion activities.
            new_phone_dur (dict): New durations for phone activities.
        """
        self.update_duration(self.dur_motion, new_motion_dur, self.motion_alert)
        self.update_duration(self.dur_phone, new_phone_dur, self.phone_alert)

    def set_alert(self, alert_dict: dict, key: int, hist_left: int, hist_right: int, duration: int) -> None:
        """Set or reset the alert status based on the duration and hysteresis thresholds.

        Args:
            alert_dict (dict): Dictionary holding the alert status.
            key (int): Key for the dictionary to update.
            hist_left (int): Hysteresis threshold for resetting the alert.
            hist_right (int): Hysteresis threshold for setting the alert.
            duration (int): Current duration value.
        """
        if not alert_dict[key] and duration == hist_right:
            alert_dict[key] = 1
        elif alert_dict[key] and duration == hist_left:
            alert_dict[key] = 0

    def update_duration(self, duration_dict: dict, new_dur: dict, alert_dict: dict) -> None:
        """Update the duration and alert dictionaries with new durations.

        Args:
            duration_dict (dict): Existing duration dictionary to update.
            new_dur (dict): New durations to add to the existing dictionary.
            alert_dict (dict): Dictionary holding the alert status.
        """
        to_delete = []

        # Update durations and alerts based on new_dur
        for key, value in new_dur.items():
            if key in duration_dict:
                duration_dict[key] += value
            else:
                duration_dict[key] = value
                alert_dict[key] = 0
            self.set_alert(alert_dict, key, self.histeresis_left, self.histeresis_right, duration_dict[key])

        # Adjust durations to be within bounds and mark absent keys for deletion
        for key in list(duration_dict.keys()):
            if key not in new_dur:
                to_delete.append(key)
            else:
                duration_dict[key] = max(0, min(self.max_value, duration_dict[key]))

        # Remove keys marked for deletion
        for key in to_delete:
            duration_dict.pop(key, None)
            alert_dict.pop(key, None)


        