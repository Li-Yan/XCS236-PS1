import torch
from submission.likelihood import log_likelihood

def classification(model, text):
    """
    Classify whether the string `text` is randomly generated or not.
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: True if `text` is a random string. Otherwise return False
    """

    with torch.no_grad():
        ## TODO: Return True if `text` is a random string. Or else return False.
        ## Hint: Your answer should be VERY SHORT! Our implementation has only 2 lines,
        ##       and yours shouldn't be longer than 7 lines. You should look at the plots
        ##       you generated in Question 6d very carefully and make use of the log_likelihood() function.
        ##       There should be NO model training involved.

        ### START CODE HERE ###
        ### END CODE HERE ###
        raise NotImplementedError