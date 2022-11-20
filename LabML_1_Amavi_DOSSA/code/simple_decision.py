
class SimpleDecision:
    def _init_(self):
        self._split_var0 = 1              # split variable, first layer
        self._split_val0 = 37.695206      # split value first layer
        self._split_val_sat = 0           # split variable for the sat. part of layer 0
        self._split_var_sat = -96.032692  # its split var
        self._split_val_not = 0           # split variable for the not sat. part of layer 0
        self._split_var_not = -112.548331 # its split var

    def predict(self, x):
        if(x[self._split_var0] >= self._split_val0):
            if(x[self._split_var_sat] >= self._split_val_sat):
                return 0
            else:
                return 1

        else:
            if(x[self._split_var_not] >= self._split_val_not):
                return 1
            else:
                return 0