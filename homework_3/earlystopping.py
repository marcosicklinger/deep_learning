import torch
import numpy as np

class EarlyStopping:
    """
    Early stopping procedure
    """

    def __init__(self, wordy = False, tolerance = .5, saving_path = 'early_stopping.pt', messenger = print):
        """
        Members
        -------
        _wordy: bool
                boolean for printing (True) or not (False) updates
        _path:  str
                string indicating where to save best model
        _messenger: function
                    function used to print class messages
        _tolerance: Number
                    maximum allowed change in generalized loss 
        _min:   Number
                variable for saving the best loss computed up to current epoch
        _generalized_loss:  Number
                            variable for saving the change in loss function from previous to current step
        steps: Number
                number of epochs completed before early stopping
        early_stop: bool
                    boolean set to `True` if threshold is reached, `False` otherwise

        Parameters
        ----------
        wordy:  bool
                boolean passed as `True` if class messages are requested, `False` otherwise
        saving_path:    str
                        string indicating a path where to save best model
        messenger:  function
                    function used to print class messages 
        """
        
        self._wordy = wordy
        self._path = saving_path
        self._messenger = messenger
        self._tolerance = tolerance
        self._min = np.Inf
        self._generalized_loss = 0
        self.steps = 0
        self.early_stop = False

    def __call__(self, loss, model):
        """
        Parameters
        ----------
        loss:   Number
                computed loss function at current epoch
        model:  Class[torch.nn.Module]
                model used for learning
        """

        self.steps += 1

        if loss < self._min:
            if self._wordy:
                self._messenger('early_stopping class message: good step - generalized loss {:.3f}'.format(self.generalized_loss(loss, self._min)))
            self._min = loss
            self.save_model(loss, model)
        elif loss >= self._min:
            gen_loss = self.generalized_loss(loss, self._min)
            if self._wordy:
                self._messenger('early_stopping class message: bad step - generalized loss {:.3f}'.format(gen_loss))
            if gen_loss > self._tolerance:
                self.early_stop = True

    def save_model(self, loss, model):
        """
        Function for saving the model

        Parameters
        ----------
        loss:   Number
                computed loss function at current epoch
        model:  Class[torch.nn.Module]
                model used for learning
        """

        torch.save(model.state_dict(), self._path)

    def generalized_loss(self, loss, Min):
        """
        Function for computing the generalized loss

        Parameters
        ----------
        loss:   Number
                computed loss function
        Min:    Number
                minimum loss function computed up to current epoch
        """
        return loss/Min-1
