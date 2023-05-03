import numpy as np
import matplotlib.pyplot as plt

class ActivitiesDataset:
    def __init__(self):
        self.activities_data = None

    def generate_activities_data(self):
        s = np.zeros(7)
        s[0:5] = 1
        for d in range(9):
            s = np.concatenate((s, s), axis=None)
        ni = np.random.randint(5, size=(1, len(s))) * 0.01
        s = s + ni

        for r in range(4):
            s = np.concatenate((s, s), axis=0)

        for r in range(15):
            s[r, :] = np.roll(s[0, :], r * 2)

        for r in range(1, 10):
            ni = np.random.randint(5 * r, size=(1, s.shape[1])) * 0.01
            s[r, :] = s[r, :] + ni

        x = np.arange(1, s.shape[1] + 1)
        y = x * 0.0001
        y = np.reshape(y, (1, s.shape[1]))

        for r in range(1, 10):
            s[r, :] = s[r, :] + y
            s[r, :] = s[r, :] * r

        self.activities_data = s[0:10, :]

    def save_data_to_csv(self, filename):
        np.savetxt(filename, self.activities_data, delimiter=',')

    def load_data_from_csv(self, filename):
        self.activities_data = np.genfromtxt(filename, delimiter=',')

    def plot_activities(self, range_from, range_to):
        for r in range(10):
            plt.plot(self.activities_data[r, range_from:range_to], label=f'Activity {r + 1}')

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

def main():
    print("running")
    activities_dataset = ActivitiesDataset()
    activities_dataset.generate_activities_data()
    activities_dataset.save_data_to_csv('activities.csv')
    activities_dataset.load_data_from_csv('activities.csv')
    activities_dataset.plot_activities(3400, 3500)

if __name__ == "__main__":
    main()
