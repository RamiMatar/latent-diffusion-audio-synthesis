class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def step_ema(self, model, ema_model, start_step = 5000):
        self.step += 1
        if self.step == start_step:
            ema_model.load_state_dict(model.state_dict())
        elif self.step < start_step:
            return
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            current_weight, ema_weight = current_param.data, ema_param.data
            ema_param.data = self.beta * ema_weight + (1 - self.beta) * current_weight
