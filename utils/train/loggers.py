import pylab as pl

def print_log(step, loss, eval_loss):
    print(f"Step {step}, loss {loss[-1]}, eval_loss {eval_loss[-1]}")


def plot_log(losses, eval_losses):
    pl.semilogy(losses)
    pl.show()
    pl.semilogy(eval_losses)
    pl.show()