import data_processing as dp
import pol_regression as pol
import matplotlib.pyplot as plt
import os

def main():
    # Data processing
    df = dp.load_data('pol_regression.csv')

    x = df['x']
    y = df['y']

    # Plot training data
    plt.figure()
    plt.xlim((-5, 5))
    plt.plot(x, y, 'o', color='g')

    colors = ['g', 'r', 'y', 'b', 'c', 'm', 'b']

    # Perform polynomial regression for powers 0 to 10
    for i, degree in enumerate([0, 1, 2, 3, 4, 5, 10]):
        w = 1

        if degree != 0:
            # Calculate the coefficients based on the training values
            w = pol.pol_regression(x, y, degree)

        y_hat = pol.prediction(x, w, degree)

        list = zip(*sorted(zip(*(x, y_hat))))
        plt.plot(*list, colors[i])

    plt.legend(('training points', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^4$', '$x^5$', '$x^{10}$'), loc = 'lower right')
    plt.savefig(os.path.join('images', 'polynomial.png'))

if __name__ == '__main__':
    main()