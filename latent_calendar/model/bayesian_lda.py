import numpy as np

import pymc as pm
import pytensor.tensor as pt


def create_logp(n_tokens: int):
    def logp_lda_doc(beta: pt.TensorVariable, theta: pt.TensorVariable):
        """Returns the log-likelihood function for given documents.

        K : number of topics in the model
        V : number of words (size of vocabulary)
        D : number of documents (in a mini-batch)

        Parameters
        ----------
        beta : tensor (K x V)
            Word distributions.
        theta : tensor (D x K)
            timeslot distributions for row.
        """

        def ll_docs_f(docs: np.ndarray):
            dixs, vixs = docs.nonzero()
            vfreqs = docs[dixs, vixs]
            ll_docs = (
                vfreqs
                * pm.math.logsumexp(
                    pt.log(theta[dixs]) + pt.log(beta.T[vixs]), axis=1
                ).ravel()
            )

            # Per-word log-likelihood times num of tokens in the whole dataset
            return pt.sum(ll_docs) / (pt.sum(vfreqs) + 1e-9) * n_tokens

        return ll_docs_f

    return logp_lda_doc


class BayesianLatentCalendar:
    def _define_model(self) -> pm.Model:
        """Define the model."""
        with pm.Model() as self.model:
            ...

    def fit(self, X, y=None) -> "BayesianLatentCalendar":
        """Just sum over all the rows."""
        ...


# n_topics = 10
# n_words = 1000
# # we have sparse dataset. It's better to have dence batch so that all words accure there
# minibatch_size = 128


# logp_lda_doc = create_logp(n_tokens)


# # defining minibatch
# doc_t_minibatch = pm.Minibatch(docs_tr.toarray(), minibatch_size)
# doc_t = pt.shared(docs_tr.toarray()[:minibatch_size])
# with pm.Model() as model:
#     theta = pm.Dirichlet(
#         "theta",
#         a=pm.floatX((1.0 / n_topics) * np.ones((minibatch_size, n_topics))),
#         shape=(minibatch_size, n_topics),
#     )
#     beta = pm.Dirichlet(
#         "beta",
#         a=pm.floatX((1.0 / n_topics) * np.ones((n_topics, n_words))),
#         shape=(n_topics, n_words),
#     )
#     # Note, that we defined likelihood with scaling, so here we need no additional `total_size` kwarg
#     doc = pm.DensityDist("doc", logp_lda_doc(beta, theta), observed=doc_t)
