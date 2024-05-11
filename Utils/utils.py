import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import ttest_rel

from Genetic.Representation import Function


class StatsHelper:
    @staticmethod
    def get_confidence(scores):
        mu = np.mean(scores) * 100
        sm = 2.78 * (np.std(scores) * 100) / np.sqrt(len(scores))
        print('Score CV: {:.2f} ({:.2f}, {:.2f})'.format(mu, mu - sm, mu + sm))

    @staticmethod
    def plot_box(folds1, name1, folds2, name2, test=ttest_rel):
        F, p = test(folds1, folds2)
        print("F = {:.2f}, p = {}".format(F, p))
        df1 = pd.DataFrame({'C-Index': folds1})
        df1['Function'] = name1
        df2 = pd.DataFrame({'C-Index': folds2})
        df2['Function'] = name2
        df = pd.concat([df1, df2])
        sns.catplot(x='Function', y='C-Index', kind="box", data=df)
        plt.title("F = {:.2f}, p = {}".format(F, p))


class FunctionHelper:
    def __init__(self, func_rep):
        self.func_rep = func_rep
        self.f = Function(self.func_rep)

    def get_function_name(self):
        return '_'.join([str(x) for x in self.func_rep])

    def plot(self, x_min=-1, x_max=1, y_min=-1, y_max=1, num_points=1000, subtitle=''):
        plt.figure()
        plot_range = np.linspace(x_min, x_max, num_points)

        values = []
        for i in plot_range:
            values.append(self.f.s2(float(i)))
        plt.plot(plot_range, values)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axvline(0, c='C1', ls='--')
        plt.axhline(0, c='C1', ls='--')

        plt.title(subtitle + ' - ' + self.get_function_name())

    def plot_surv_loss(self, x_min=-1, x_max=1, y_min=-1, y_max=1, num_points=1000, subtitle=''):
        plot_range = np.linspace(x_min, x_max, num_points)

        e_values = []
        c_values = []
        for i in plot_range:
            e_values.append(self.f.sl(float(i)))
            c_values.append(self.f.sr(float(i)))

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(plot_range, e_values)
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_ylim(y_min, y_max)
        ax[0].axvline(0, c='C1', ls='--')
        ax[0].axhline(0, c='C1', ls='--')
        ax[0].set_title('Events Loss')

        ax[1].plot(plot_range, c_values)
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_ylim(y_min, y_max)
        ax[1].axvline(0, c='C1', ls='--')
        ax[1].axhline(0, c='C1', ls='--')
        ax[1].set_title('Censored Loss')

        fig.suptitle(subtitle + ' - ' + self.get_function_name())


class Experiment:
    def __init__(self, model_class, ds, loss_f, step_f, use_clb, a=1, b=1, batch_size=128, epochs=500, patience=100, verbose=False):
        self.model_class = model_class
        self.ds = ds
        self.loss_f = loss_f
        self.step_f = step_f
        self.use_clb = use_clb
        self.a = a
        self.b = b

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose

    def fit(self, test_id=0, val_id=1):

        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = self.ds.get_train_val_test_from_splits(test_id=test_id,
                                                                                        val_id=val_id)

        optimizer = 'adam'
        # optimizer = Adam(lr=0.0001)
        print('Train Shape:', x_train.shape)
        print('Val Shape:', x_val.shape)
        print('Test Shape:', x_test.shape)

        xy = (x_train, ye_train), (x_val, ye_val), (x_test, ye_test)

        # file_name = get_function_name(self.func_rep) + '_' + self.ds.get_dataset_name() + f'_test_id_{self.test_id}_val_id_{self.val_id}.h5'
        callbacks = [EarlyStopping(monitor='val_cindex', patience=self.patience, restore_best_weights=True, verbose=1, mode='max')]

        model = self.model_class(xy, f_loss=self.loss_f, f_step=self.step_f, use_clb=self.use_clb, a=self.a, b=self.b, sample_size=-1)
        model.create_model()
        model.set_config(optimizer=optimizer, batch_size=self.batch_size, epochs=self.epochs,
                         test_mode=True, callbacks=callbacks, verbose=self.verbose)
        model.fit()
        self.hist = model.history.history
        # print('Max Train C-Index', np.max(model.history.history['cindex']))
        # print('Max Val C-Index', np.max(model.history.history['val_cindex']))

        ci_test = model.evaluate_test()
        ci_val = model.evaluate(x_val, ye_val)
        print(f'Val Id {val_id} -Val C-Index: {ci_val}')
        print(f'Test Id {test_id} -Test C-Index: {ci_test}')

        return model, ci_test

    def get_results_cv(self, folds_ids=None):

        if folds_ids is None:
            folds_ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        cfn_folds = []
        hist = []
        for test_id, val_id in folds_ids:
            (x_train, ye_train, y_train, e_train,
             x_val, ye_val, y_val, e_val,
             x_test, ye_test, y_test, e_test) = self.ds.get_train_val_test_from_splits(test_id=test_id,
                                                                                  val_id=val_id)

            optimizer = 'adam'
            # optimizer = Adam(lr=0.0001)

            xy = (x_train, ye_train), (x_val, ye_val), (x_test, ye_test)

            # file_name = get_function_name(self.func_rep) + '_' + self.ds.get_dataset_name() + f'_test_id_{self.test_id}_val_id_{self.val_id}.h5'
            callbacks = [EarlyStopping(monitor='val_cindex', patience=self.patience, restore_best_weights=True, verbose=1, mode='max')]

            model = self.model_class(xy, f_loss=self.loss_f, f_step=self.step_f, use_clb=self.use_clb, a=self.a, b=self.b, sample_size=-1)
            model.create_model()
            model.set_config(optimizer=optimizer, batch_size=self.batch_size, epochs=self.epochs,
                             test_mode=True, callbacks=callbacks, verbose=self.verbose)
            model.fit()
            hist.append(model.history.history)
            print('Max Train C-Index', np.max(model.history.history['cindex']))
            print('Max Val C-Index', np.max(model.history.history['val_cindex']))

            ci_test = model.evaluate_test()
            print(f'Fold {test_id} -Test C-Index: {ci_test}')
            cfn_folds.append(ci_test)
        print('Folds C-Index:')
        print(cfn_folds)
        StatsHelper.get_confidence(cfn_folds)
        self.folds_scores = cfn_folds
        self.folds_hists = hist
        return StatsHelper.get_confidence(self.folds_scores)

    @staticmethod
    def plot_hist(hist, name):
        plt.plot(hist['cindex'], label='cindex')
        plt.plot(hist['val_cindex'], label='cindex_val')
        plt.title(name)
        plt.legend()

    def plot_hists(self, hists, name='Training History'):
        for h in hists:
            plt.figure()
            self.plot_hist(h, name)




