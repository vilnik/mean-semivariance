import json
import os
import logging

logger = logging.getLogger(__name__)

class ProblemSpec:
    def __init__(self, spec_name):
        logger.info(f"Getting problem specification for {spec_name}.")
        self.set_params(spec_name)
        self.check_params()
        self.short_name = self.get_short_name()
        self.long_name = self.get_long_name()

    def set_params(self, spec_name):
        # Assuming the JSON file is in a directory named config at the same level as the src directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(base_dir, 'config', 'problem_specs_params.json')
        with open(json_path, 'r') as file:
            all_params = json.load(file)
        self.params = all_params.get(spec_name, None)

        if self.params is None:
            logger.error(f"Could not find parameters corresponding to {spec_name} in problem_specs_params.json.")
            raise Exception(f"Could not find parameters corresponding to {spec_name} in problem_specs_params.json.")

    def get_params(self):
        return self.params

    def check_params(self):
        if not self.params["up_ret"] > self.params["down_ret"]:
            logger.error("up_ret must be greater than down_ret.")
            raise ValueError("up_ret must be greater than down_ret.")

        if not self.params["down_ret"] < self.params["interest"]:
            logger.error("down_ret must be less than interest.")
            raise ValueError("down_ret must be less than interest.")

        if not self.params["up_ret"] > self.params["interest"]:
            logger.error("up_ret must be greater than interest.")
            raise ValueError("up_ret must be greater than interest.")

        if not self.params["up_prob"] > 0:
            logger.error("up_prob must be greater than 0.")
            raise ValueError("up_prob must be greater than 0.")

        if not self.params["shock_prob"] >= 0:
            logger.error("shock_prob must be at least 0.")
            raise ValueError("shock_prob must be at least 0.")

        if not 1 - self.params["up_prob"] - self.params["shock_prob"] > 0:
            logger.error("The sum of up_prob and shock_prob must be less than 1.")
            raise ValueError("The sum of up_prob and shock_prob must be less than 1.")

    def get_short_name(self):
        if self.params is None:
            return "params not found"
        params = self.params
        period_type = "multi_period" if params["time_T"] > 1 else "one_period"
        interest_type = "interest" if params["interest"] > 0 else "no_interest"
        shock_type = "no_shocks"
        if params["shock_prob"] > 0:
            shock_type = "pos_shocks" if params["shock_ret"] > 0 else "neg_shocks"
        gamma_type = "large_gamma" if params["risk_aversion"] > 2 else "small_gamma"
        return f"{period_type}_{interest_type}_{shock_type}_{gamma_type}"

    def get_long_name(self):
        if self.params is None:
            return "params not found"
        long_name_parts = []
        for key, value in self.params.items():
            if value is None:
                value_str = 'None'
            else:
                value_str = str(value)
            long_name_parts.append(f"{key}_{value_str}")
        return '_'.join(long_name_parts)
