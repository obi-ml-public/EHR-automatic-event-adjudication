# coding: utf8
from __future__ import unicode_literals
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta, compile_infix_regex
from spacy.symbols import ORTH

__version__ = get_model_meta(Path(__file__).parent)['version']


def read_abbreviations():
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources
    from . import abbreviations  # relative-import the *package* containing the templates
    abbrevs = []
    with pkg_resources.open_text(abbreviations, 'medical_abbreviations_curated.txt') as f:
        abbrevs += [line.rstrip('\n') for line in f]
    with pkg_resources.open_text(abbreviations, 'medical_abbreviations_wiki.txt') as f:
        abbrevs += [line.rstrip('\n') for line in f]
    return abbrevs


def load(**overrides):
    nlp = load_model_from_init_py(__file__, **overrides)
    abbrevs = read_abbreviations()

    # exclusions based on common medical abbreviations (don't split these)
    # this matches with capitalization
    exclusions_cased = {abbreviation: [{ORTH: abbreviation}] for abbreviation in abbrevs}
    for k, excl in exclusions_cased.items():
        nlp.tokenizer.add_special_case(k, excl)
    # this matches any lower case tokens
    exclusions_uncased = {abbreviation.lower(): [{ORTH: abbreviation.lower()}] for abbreviation in abbrevs}
    for k, excl in exclusions_uncased.items():
        try:
            nlp.tokenizer.add_special_case(k, excl)
        except:
            print('failed to add exception: {}'.format(k))

    # additional rules for more fine-grained splitting of tokens
    # do split tokens when encountering symbols like ():= without whitespace
    infixes = nlp.Defaults.infixes + tuple(r'''[\(\)\[\]:="]''')
    infix_regex = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer

    return nlp
