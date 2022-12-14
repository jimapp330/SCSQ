#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import  os
import _pickle as pkl
import copy
from lib.shap_utils_p import *
from torch.utils.data import TensorDataset
from lib.BasicTrainer_sw import Trainer


class DShap(object):
    def __init__(self, model, args, train_set,test_set,heldout_set, num_test, sources=None,
                 sample_weight=None, directory=None, problem='classification',
                 model_family='Attn_LSTM', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        # if seed is not None:
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        self.args = args
        self.heldout_set=heldout_set
        self.train_set = train_set
        self.learning_rate=0.001
        self.test_set = test_set
        self.problem = problem
        self.max_epochs=20
        self.model_family = model_family
        self.metric = metric
        self.criterion=torch.nn.CrossEntropyLoss()
        self.directory = directory
        self.batch_size = 32
        self.args.topk = 100
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(train_set,test_set, num_test,
                                      sources, sample_weight)

        self.model = model.to(args.device)

        self.device='cuda'
        self.optimizer='adam'
        self.random_score = self.init_score(self.metric)




    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        # test_loader = DataLoader(dataset=self.test_set,
        #                          batch_size=1,
        #                          shuffle=False)
        y_t=[]
        for batch_idx, (data, target) in enumerate(self.test_set):
            y_t.extend(target.detach().cpu().numpy())
        y_test=np.array(y_t)
        # if metric == 'accuracy':
        #     hist = np.bincount(y_test).astype(float)/len(y_test)
        return np.max(y_test)

    def _initialize_instance(self, train_set,test_set, num_test,
                             sources=None, sample_weight=None):
        if sources is None:
            self.sources = {i:np.array([i]) for i in range(len(train_set))}
        elif not isinstance(sources, dict):
            self.sources = {i:np.where(sources==i)[0] for i in set(sources)}
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        n_sources = len(train_set)
        n_points =len(train_set)
        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number,
            n_points, n_sources)

    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                    n_points, n_sources):
        tmc_dir = os.path.join(
            directory,
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory,
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g},
                 open(g_dir, 'wb'))

    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                   for name in previous_results if 'mem_g' in name]
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0'
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0'
        return tmc_number, g_number
    def value(self, model,test_set, metric=None):
        if metric is None:
            metric = self.metric
        if metric == 'accuracy':
            return  score(model,test_set,self.device)

    def run(self, save_every, counts, tolerance=0.01, g_run=True, tmc_run = False, loo_run=True):
        # if loo_run:
        #     try:
        #         len(self.vals_loo)
        #     except:
        #         self.vals_loo = self._calculate_loo_vals(sources=self.sources)
        #         self.save_results(overwrite=True)
        # print('LOO values calculated!')

        while tmc_run or g_run:
            if g_run:
                if error(self.mem_g) >= counts:
                    g_run = False
                else:
                    self._g_shap(save_every, learning_rate=self.args.lr_init, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                if error(self.mem_tmc) > counts:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every,
                        learning_rate=self.args.lr_init,
                        tolerance=tolerance,
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
            if self.directory is not None:
                self.save_results()
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory,
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory,
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g},
                 open(g_dir, 'wb'))

    def _calculate_loo_vals(self, sources=None, metric=None):
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.train_set))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric
        model, optimizer, train_losses=fit(self.train_set,copy.deepcopy(self.model),self.max_epochs,
                                           self.criterion,self.optimizer,self.device,self.learning_rate)
        baseline_value = self.value(model,self.test_set, metric=metric)
        vals_loo = np.zeros(len(self.train_set))
        for i in sources.keys():
            x=[]
            y=[]
            index=[]
            for j in range(len(self.train_set)):
                if j!=sources[i]:
                    x.append(self.train_set[j][0].numpy())
                    y.append(self.train_set[j][1])
                    index.append((self.train_set[j][2]))
            train_set_new = TensorDataset(torch.Tensor(np.array(x)),torch.LongTensor(np.array(y)),torch.LongTensor(np.array(index)))
            model, optimizer, train_losses=fit(train_set_new,copy.deepcopy(self.model),self.max_epochs,
                                           self.criterion,self.optimizer,self.device,self.learning_rate)

            removed_value = self.value(model,copy.deepcopy(self.test_set), metric=metric)
            vals_loo[sources[i]] = (baseline_value.cpu() - removed_value.cpu())
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo

    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        #model = copy.deepcopy(self.model)
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.train_set))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        address = None
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        for iteration in range(iterations):
            model_g = copy.deepcopy(self.model)
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(
                    iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            history =fit_gshape(self.args,model_g,self.train_set,self.test_set,1,
                                self.criterion,self.optimizer,learning_rate,self.device)
            val_result = history['metrics']
            # marginal_contribs[1:] = marginal_contribs[1:]+ val_result[0][1:]
            # marginal_contribs[1:] = marginal_contribs[1:]- val_result[0][:-1]
            marginal_contribs[1:] = np.diff(val_result[0])
            # individual_contribs = np.zeros(len(self.train_set))
            # for i, index in enumerate(history['idxs'][0]):
            #     individual_contribs[sources[i]] += marginal_contribs[i]
            #     individual_contribs[sources[i]] /= len(sources[i])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(marginal_contribs, (1, -1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(history['idxs'][0], (1, -1))])
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            accs = []
            for _ in range(10):
                model = copy.deepcopy(self.model)
                model, optimizer1, train_losses = fit(self.train_set,model,1,
                                           self.criterion,'adam',self.device,10**(-i))
                accs.append(score(model,self.test_set,self.device).item())
            if np.mean(np.array(accs)) - np.std(np.array(accs)) > best_acc:
                best_acc  = np.mean(np.array(accs)) - np.std(np.array(accs))
                learning_rate = 10**(-i)
        return learning_rate
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.train_set))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance
        marginals, idxs = [], []
        for iteration in range(iterations):
            model = copy.deepcopy(self.model)
            if 10 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.one_iteration(model,
                tolerance=tolerance,
                sources=sources
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc,
                np.reshape(marginals, (1, -1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc,
                np.reshape(idxs, (1, -1))
            ])

    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        for _ in range(1):
            model, optimizer, train_losses = fit(self.train_set,copy.deepcopy(self.model),self.max_epochs,
                                           self.criterion,self.optimizer,self.device,self.learning_rate)
            for _ in range(10):

                scores.append(score(model,self.test_set,self.device
                ).cpu())
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)

    def one_iteration(self, model, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.rgs.device)
        else:
            raise ValueError

        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.train_set))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        marginal_contribs = np.zeros(len(self.train_set))
        index = np.zeros(len(self.train_set))
        truncation_counter = 0

        # x_stack = torch.zeros((1,) + tuple(self.train_set[0][0].shape))#??????
        # y_stack = torch.zeros((1,),dtype= int)#??????
        x_stack = torch.zeros((0, self.train_set[1][0].shape[1], self.train_set[1][0].shape[2]))  # ??????
        y_stack = torch.zeros((0, self.train_set[1][1].shape[1]))  # ??????
        new_score = torch.tensor(self.random_score)

        # train_loader = DataLoader(dataset=self.train_set,
        #                           batch_size=1,
        #                           shuffle=True)

        i=0
        for idx, (x,y) in enumerate(self.train_set):
            x_stack = x_stack.to('cuda')
            x = x.to('cuda')
            y_stack = y_stack.to('cuda')
            y = y.to('cuda')
            index[i]=idx
            old_score = new_score
            # x_stack=x_stack.permute((1, 0, 2, 3))
            # x_stack = torch.column_stack([x_stack, x]).permute((1,0,2,3))
            x_stack = torch.column_stack([x_stack, x])
            y_stack = torch.column_stack([y_stack, y])
            for epoch in range(0, 1):#self.max_epochs
                self.optimizer.zero_grad()

                x_stack = x_stack.to(self.device)
                model = model.to(self.device)
                y_stack = y_stack.to(self.device)

                # Forward pass
                if self.args.model == 'Autoformer':
                    y_true = y_stack[:, :self.args.label_len + self.args.window * self.args.horizon]
                else:
                    y_true = y_stack[:, :self.rgs.window * self.args.horizon]

                if self.args.model in ['informer', 'Autoformer']:
                    label = y_true[:, -self.args.window * self.args.horizon:, 0]
                else:
                    label = y_true[:, -self.args.window * self.args.horizon:]

                output, _ = model(x_stack, y_true, teacher_forcing_ratio=0.)

                loss = loss_fun(output.squeeze().cuda(), label)
                # Backward pass
                loss.backward()
                self.optimizer.step()
            new_score = score(self.args,self.model,self.test_set,self.device)
            marginal_contribs[sources[i]] = (new_score.cpu() - old_score.cpu())
            marginal_contribs[sources[i]] /= len(sources[i])
            i+=1
            distance_to_full_score = np.abs(new_score.cpu() - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0

        return marginal_contribs,index

    def merge_results(self, max_samples=None):
        """Merge all the results from different runs.

        Returns:
            combined marginals, sampled indexes and values calculated
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        # if self.model_family not in ['logistic', 'NN']:
        #     return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    def _merge_parallel_results(self, key, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory)
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, len(self.train_set)))
        n_sources = len(self.train_set)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.train_set))
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory,
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory,
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs},
                 open(merged_dir, 'wb'))
        return mem, idxs, vals


    def _portion_performance_series(self, idxs, scaler):
        """Given a set of indexes, starts removing points from
        the first elemnt and evaluates the new model after
        removing each point."""

        top_k = idxs[:self.args.topk]
        heldout_index = list(set(idxs).difference(set(top_k)))
        #heldout_index = np.array(top_k)
        train_set = self.train_set.dataset
        data = torch.utils.data.Subset(train_set, heldout_index)
        train_loader = DataLoader(dataset=data,
                                  batch_size=32,
                                  shuffle=True)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr_init, weight_decay=0)
        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        # learning rate decay
        lr_scheduler = None
        if self.args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(self.args.lr_decay_step.split(','))]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                milestones=lr_decay_steps,
                                                                gamma=self.args.lr_decay_rate)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

        # start training
        val_loader = DataLoader(dataset=self.heldout_set.dataset,
                                  batch_size=32,
                                  shuffle=True)

        trainer = Trainer('normal', self.model, loss, optimizer, train_loader, val_loader, self.test_set, scaler,
                          self.args, lr_scheduler=lr_scheduler)

        if self.args.mode == 'Train':
            trainer.train()
            return trainer.results
