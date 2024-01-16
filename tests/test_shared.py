def get_default_chess_database_path():
    # The database is far too large to ship with the repo; you can grab it on the lichess website:
    # https://database.lichess.org/
    # This will use the November 2023 database, which is the most recent one as of this writing.
    return 'chess_training/lichess_db_standard_rated_2023-11.pgn.zst'


def get_default_model_configurations():
    # The tuple contains the model size (in the name), plus the accumulation steps to use for each size.
    # return [('70m', 2), ('160m', 4), ('410m', 4), ('1b', 8), ('1.4b', 16), ('2.8b', 32)]
    return [('70m', 2), ('160m', 4), ('410m', 4), ('1b', 8)]
