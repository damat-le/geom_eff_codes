from pytorch_lightning import data_loader
## Utils to handle newer PyTorch Lightning changes from version 0.6
def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """
    def func_wrapper(self):
        try: # Works for version 0.6.0
            return data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)
    return func_wrapper
