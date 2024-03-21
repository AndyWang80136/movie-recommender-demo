from movie_recommender.inference import RecallAlgo, get_recall_algo_params


def test_get_recall_algo_params():
    algo_params = get_recall_algo_params(algo_url='http://localhost',
                                         algo_params={
                                             'itemcf-ratings':
                                             dict(similarity_top_k=10,
                                                  output_top_k=100),
                                             'usercf-ratings':
                                             dict(similarity_top_k=[10, 100],
                                                  output_top_k=100)
                                         })
    assert isinstance(algo_params, dict) and all(
        isinstance(param, RecallAlgo) for param_list in algo_params.values()
        for param in param_list)
    assert len(algo_params['itemcf-ratings']) == 1
    assert len(algo_params['usercf-ratings']) == 2
    assert algo_params['itemcf-ratings'][
        0].url == 'http://localhost/itemcf-ratings'
    assert algo_params['itemcf-ratings'][0].algo == 'itemcf-ratings'
    assert algo_params['itemcf-ratings'][0].hyperparams == algo_params[
        'usercf-ratings'][0].hyperparams == dict(similarity_top_k=10,
                                                 output_top_k=100)
    assert algo_params['usercf-ratings'][1].hyperparams == dict(
        similarity_top_k=100, output_top_k=100)
