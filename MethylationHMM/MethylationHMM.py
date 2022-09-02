
import numpy as np
import pandas as pd
from .my_Base import BaseHMM
from pyhhmm.utils import check_if_attributes_set, normalise
from scipy.stats import binom


class MethylationHMM(BaseHMM):
    """Hidden Markov Model with binomial emission.

    :param n_states: number of hidden states in the model
    :type n_states: int
    :param n_emissions: number of distinct observation symbols per state
    :type n_emissions: int
    :param n_features: list of the number of possible observable symbols for each emission
    :type n_features: list
    :param tr_params: controls which parameters are updated in the training process; can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, and other characters for subclass-specific emission parameters, defaults to 'ste'
    :type tr_params: str, optional
    :param init_params: controls which parameters are initialised prior to training.  Can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, and other characters for subclass-specific emission parameters, defaults to 'ste'
    :type init_params: str, optional
    :param init_type: name of the initialisation  method to use for initialising the start, transition and emission matrices, defaults to 'random'
    :type init_type: str, optional
    :param pi_prior: float or array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for 'pi', defaults to 1.0
    :type pi_prior: float or array_like, optional
    :param pi: array of shape (n_states, ) giving the initial state occupation distribution 'pi'
    :type pi: array_like
    :param A_prior: float or array of shape (n_states, n_states), giving the parameters of the Dirichlet prior distribution for each row of the transition probabilities 'A', defaults to 1.0
    :type A_prior: float or array_like, optional
    :param A: array of shape (n_states, n_states) giving the matrix of transition probabilities between states
    :type A: array_like
    :param B: the probabilities of emitting a given symbol when in each state
    :type B: list
    :param missing: a value indicating what character indicates a missed observation in the observation sequences, defaults to np.nan
    :type missing: int or np.nan, optional
    :param nr_no_train_de: this number indicates the number of discrete  emissions whose Matrix Emission Probabilities are fixed and are not trained; it is important to to order the observed variables such that the ones whose emissions aren't trained are the last ones, defaults to 0
    :type nr_no_train_de: int, optional
    :param state_no_train_de: a state index for nr_no_train_de which shouldn't be updated; defaults to None, which means that the entire emission probability matrix for that discrete emission will be kept unchanged during training, otherwise the last state_no_train_de states won't be updated, defaults to None
    :type state_no_train_de: int, optional
    :param learning_rate: a value from [0,1), controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    :param verbose: flag to be set to True if per-iteration convergence reports should be printed, defaults to True
    :type verbose: bool, optional
    """

    def __init__(
        self,
        n_states,
        distance_param = 75,
        tr_params='e',
        init_params='e',
        init_type='random',
        pi_prior=1.0,
        A_prior=1.0,
        missing=np.nan,
        nr_no_train_de=0,
        state_no_train_de=None,
        learning_rate=0.1,
        verbose=True,
    ):
        """Constructor method

        :raises ValueError: if init_type is not one of ('uniform', 'random')
        """

        super().__init__(
            n_states,
            tr_params=tr_params,
            init_params=init_params,
            init_type=init_type,
            pi_prior=pi_prior,
            A_prior=A_prior,
            verbose=verbose,
            learning_rate=learning_rate,
        )
        self.n_emissions = 1
        self.n_features = [2]
        self.MISSING = missing
        self.nr_no_train_de = nr_no_train_de
        self.state_no_train_de = state_no_train_de
        
        self.dist_param = distance_param
        self.A = self._transition_matrix

    def __str__(self):
        """Function to allow directly printing the object.
        """
        temp = super().__str__()
        return temp + '\nB:\n' + str(self.B)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def missing(self):
        """Getter for the missing value character in the data. 
        """
        return self.MISSING

    @missing.setter
    def missing(self, value):
        """Setter for the missing value character in the data. 

        :param value: a value indicating what character indicates a missed
            observation in the observation sequences
        :type value: int or np.nan
        """
        self.MISSING = value

    def get_n_fit_scalars_per_param(self):
        """Return a mapping of trainable model parameters names (as in ``self.tr_params``) to the number of corresponding scalar parameters that will actually be fitted.
        """
        ns = self.n_states
        ne = self.n_emissions
        nf = self.n_features
        nnt = self.nr_no_train_de
        return {
            's': ns,
            't': ns * ns,
            'e': sum(ns * (nf[i] - 1) for i in range(ne - nnt)) if self.state_no_train_de is None else sum(ns * (nf[i] - 1) for i in range(ne - nnt) if i != self.state_no_train_de),
        }

    
    def sample(self, n_sequences=1, n_samples=[1], depth=[0], positions=[0], return_states=False): # Change because of A
        """Generate random samples from the model.

        :param n_sequences: number of sequences to generate, defeaults to 1.
        :type n_sequences: int, optional
        
        :param n_samples: number of samples to generate per sequence; defeaults to 1. If multiple
                sequences have to be generated, it is a list of the individual
                sequence lengths
        :type n_samples: list, optional
        
        :param depth: number of trials to generate per sample; defeaults to 0. If multiple
                sequences have to be generated, it is a list of (n_sequences x [n_samples])
        :type depth: int, optional
        
        :param return_states: if True, the method returns the state sequence from which the samples
            were generated, defeaults to False.
        :type return_states: bool, optional
        
        :return: a list containing one or n_sequences sample sequences
        :rtype: list
        
        :return: a list containing the state sequences that
                generated each sample sequence
        :rtype: list
        """
        samples = []
        state_sequences = []

        startprob_cdf = np.cumsum(self.pi)
        
        if n_sequences > 1:
            assert len(n_samples) == n_sequences, f'Len of list of n_samples should be equal to {n_sequences}'
            for i_seq in range(n_sequences):
                if n_samples[i_seq] > 1:
                    assert len(depth[i_seq]) == n_samples[i_seq], f'Len of sublist of depth[{i_seq}] should be equal to {n_samples[i_seq]}'
                    assert len(positions[i_seq]) == n_samples[i_seq], f'Len of sublist of positions[{i_seq}] should be equal to {n_samples[i_seq]}'
                    dist_plus = np.concatenate([[100*self.dist_param],positions[i_seq][1:] - positions[i_seq][:-1]])
                    dist_minus = np.concatenate([positions[i_seq][1:] - positions[i_seq][:-1],[100*self.dist_param]])
                    currstate = (startprob_cdf > np.random.rand()).argmax()
                    state_sequence = [currstate]
                    X = [[self._generate_sample_from_state(currstate, depth[i_seq][0]),depth[i_seq][0],dist_plus[0],dist_minus[0]]]
                    for t in range(1,len(depth[i_seq])):
                        transmat_cdf = np.cumsum(self.A(dist_plus[t]), axis=1)
                        currstate = (transmat_cdf[currstate] > np.random.rand()).argmax()
                        state_sequence.append(currstate)
                        X.append([self._generate_sample_from_state(currstate, depth[i_seq][t]),depth[i_seq][t],dist_plus[t],dist_minus[t]])
                    samples.append(np.vstack(X))
                    state_sequences.append(state_sequence)
                else:
                    assert type(depth[i_seq]) is int, f'{depth[i_seq]} is not "int" but n_samples = 1'
                    assert type(positions[i_seq]) is int, f'{positions[i_seq]} is not "int" but n_samples = 1'
                    currstate = (startprob_cdf > np.random.rand()).argmax()
                    state_sequence = [currstate]
                    X = [[self._generate_sample_from_state(currstate, depth[i_seq]),depth[i_seq],100*self.dist_param,100*self.dist_param]]
                    samples.append(np.vstack(X))
                    state_sequences.append(state_sequence)
        else:
            assert type(n_samples) is int, f'{n_samples} is not "int" but n_sequences = 1'
            if n_samples > 1:
                assert len(depth) == n_samples, f'Len of sublist of depth should be equal to {n_samples}'
                assert len(positions) == n_samples, f'Len of sublist of positions should be equal to {n_samples}'
                dist_plus = np.concatenate([[100*self.dist_param],positions[1:] - positions[:-1]])
                dist_minus = np.concatenate([positions[1:] - positions[:-1],[100*self.dist_param]])
                currstate = (startprob_cdf > np.random.rand()).argmax()
                state_sequence = [currstate]
                X = [[self._generate_sample_from_state(currstate, depth[0]), depth[0],dist_plus[0],dist_minus[0]]]
                for t in range(1,len(depth)):
                    transmat_cdf = np.cumsum(self.A(dist_plus[t]), axis=1)
                    currstate = (transmat_cdf[currstate] > np.random.rand()).argmax()
                    state_sequence.append(currstate)
                    X.append([self._generate_sample_from_state(currstate, depth[t]),depth[t],dist_plus[t],dist_minus[t]])
                samples.append(np.vstack(X))
                state_sequences.append(state_sequence)
            else:
                assert type(depth) is int, f'{depth} is not "int" but n_samples = 1'
                assert type(positions) is int, f'{positions} is not "int" but n_samples = 1'
                currstate = (startprob_cdf > np.random.rand()).argmax()
                state_sequence = [currstate]
                X = [[self._generate_sample_from_state(currstate, depth),depth,100*self.dist_param,100*self.dist_param]]
                samples.append(np.vstack(X))
                state_sequences.append(state_sequence)

        if return_states:
            return samples, state_sequences
        return samples
    
    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _transition_matrix(self, dist):
        d = abs(dist)
        base_trans_mat = np.ones((self.n_states,self.n_states))/self.n_states
        same_state_mat = np.eye(self.n_states)*(self.n_states-1)/self.n_states*np.exp(-d/self.dist_param)
        change_state_mat = (np.ones((self.n_states,self.n_states))-np.eye(self.n_states))/self.n_states*np.exp(-d/self.dist_param)
        return base_trans_mat + same_state_mat - change_state_mat
    
    def _init_model_params(self):
        """Initialises model parameters prior to fitting. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._init_model_params()

        if 'e' in self.init_params:
            if self.init_type == 'uniform':
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.full(
                            (self.n_states, self.n_features[i]), 1.0 / self.n_features[i])
                        for i in range(self.n_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr='e')
            else:
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.random.rand(self.n_states, self.n_features[i])
                        for i in range(self.n_emissions)
                    ]
                    for i in range(self.n_emissions):
                        normalise(self.B[i], axis=1)

                else:
                    check_if_attributes_set(self, attr='e')
                    
        init = 1.0 / self.n_states
        self.pi = np.full(self.n_states, init)
                    
                    

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step. Extends the base classes method by adding the emission probability matrix. See _BaseHMM.py for more information.
        """
        stats = super()._initialise_sufficient_statistics()

        stats['B'] = {
            'numer': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
        }

        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs_stats, obs_seq
    ):
        """Updates sufficient statistics from a given sample. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._accumulate_sufficient_statistics(
            stats, obs_stats
        )

        if 'e' in self.tr_params:
            B_new = self._reestimate_B(
                obs_seq, obs_stats['gamma'])
            for i in range(self.n_emissions):
                stats['B']['numer'][i] += B_new['numer'][i]
                stats['B']['denom'][i] += B_new['denom'][i]

    def _reestimate_B(self, obs_seq, gamma):
        """Re-estimation of the emission matrix (part of the 'M' step of Baum-Welch). Computes B_new = expected # times in state s_j with symbol v_k /expected # times in state s_j

        :param obs_seq: array of shape (n_samples, n_features)
                containing the observation samples
        :type obs_seq: np.array
        :param gamma: array of shape (n_samples, n_states)
        :return: the modified parts of the emission matrix
        :rtype: dict
        """
        B_new = {
            'numer': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
        }

        for e in range(self.n_emissions):
            for j in range(self.n_states):
                for k in range(self.n_features[e]):
                    numer = 0.0
                    denom = 0.0
                    for t, obs in enumerate(obs_seq):
                        for sub_sample in [1]*obs[0]+[0]*(obs[1]-obs[0]):
                            if sub_sample == k:
                                numer += gamma[t][j]
                            denom += gamma[t][j]
                    B_new['numer'][e][j][k] = numer
                    B_new['denom'][e][j][k] = denom

        return B_new

    def _M_step(self, stats):
        """Performs the 'M' step of the Baum-Welch algorithm. Extends the base classes method. See _BaseHMM.py for more information.
        """
        new_model = super()._M_step(stats)

        if 'e' in self.tr_params:
            new_model['B'] = [(stats['B']['numer'][i] / stats['B']['denom'][i]) for i in range(self.n_emissions)]

        return new_model

    def _update_model(self, new_model):
        """ Updates the emission probability matrix. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._update_model(new_model)

        if 'e' in self.tr_params:
            if self.state_no_train_de is None:
                for i in range(self.n_emissions - self.nr_no_train_de):
                    self.B[i] = (1 - self.learning_rate) * new_model['B'][
                        i
                    ] + self.learning_rate * self.B[i]
            else:
                for i in range(self.n_d_emissions):
                    if i < self.n_d_emissions - self.nr_no_train_de:
                        self.B[i] = (1 - self.learning_rate) * new_model['B'][
                            i
                        ] + self.learning_rate * self.B[i]
                    else:
                        self.B[i][: -self.state_no_train_de, :] = (
                            (1 - self.learning_rate)
                            * new_model['B'][i][: -self.state_no_train_de, :]
                            + self.learning_rate *
                            self.B[i][: -self.state_no_train_de, :]
                        )

            for i in range(self.n_emissions):
                normalise(new_model['B'][i], axis=1)
                new_model['B'][i] = pd.DataFrame(new_model['B'][i]).sort_values(1).values

    def _map_B(self, obs_seq):
        """Required implementation for _map_B. Refer to _BaseHMM for more details.
        """
        B_map = np.ones((self.n_states, len(obs_seq)))

        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                assert len(obs) == 4, f'Wrong obs_seq!!! {t},{obs}'
                counts, depth, _,_ = obs
                B_map[j][t] *= binom.pmf(counts,depth,self.B[0][j][1])**np.log10(depth+2)
        return B_map

    def _generate_sample_from_state(self, state, depth):
        """Required implementation for _generate_sample_from_state. Refer to _BaseHMM for more details.
        """

        res = np.random.binomial(depth, self.B[0][state, 1])
        return np.asarray(res)
