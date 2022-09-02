import pyhhmm.utils as hu

def pretty_print_hmm(model, hmm_type='Methylation', states=None, emissions=None):
    """
    Function to pretty print the parameters of an hmm model.

    :param model: and HMM object
    :type model: object
    :param hmm_type: the type of the HMM model; can be 'Multinomial',
                'Gaussian' or 'Heterogeneous', defaults to 'Multinomial'
    :type hmm_type: str, optional
    :param states: list with the name of states, if any, defaults to None
    :type states: list, optional
    :param emissions: list of the names of the emissions, if any, defaults to None
    :type emissions: list, optional
    """
    if states is None:
        states = ['S_' + str(i) for i in range(model.n_states)]

    if emissions is None:
        emissions = create_emissions_name_list(model, hmm_type)

    hu.print_startprob_table(model, states)
    print_transition_table(model, states)

    if hmm_type == 'Methylation':
        hu.print_emission_table(model, states, emissions)
    return

def print_transition_table(model, states):
    """
    Helper method for the pretty print function. Prints the state
    transition probabilities.
    """
    print('Transitions')
    rows = []
    for i, row in enumerate(model.A(0)):
        rows.append(
            [states[i]]
            + [
                'P({}|{})={:.3f}'.format(states[j], states[i], tp)
                for j, tp in enumerate(row)
            ]
        )
    hu.print_table(rows, ['_'] + states)
    return