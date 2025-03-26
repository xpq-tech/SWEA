import sys
import os
import torch

from .queryexecutor import QueryExecutor
from SWEAOS import SWEAOSHyperParams, apply_SWEAOS_to_model
from utils.globals import KV_DIR
from rome import ROMEHyperParams, apply_rome_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from grace import GraceHyperParams, apply_grace_to_model
from utils import nethook

class ModelEditor:

    def __init__(self, query_executor):
        self._query_executor = query_executor
        self._model = self._query_executor.get_model()
        self._tokenizer = self._query_executor.get_tokenizer()
        self._model_name = self._query_executor.get_model_name()
        self._model_device = self._query_executor.get_device()

    def edit_model(self, fact):
        raise NotImplementedError()  # Override in concrete classes

    def restore_model(self):
        raise NotImplementedError()  # Override in concrete classes


class InContextModelEditor(ModelEditor):

    def __init__(self, query_executor: QueryExecutor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        context = 'Imagine that ' + fact.get_fact_phrased() + '\n'
        print(f'In Context Editing added context: {context}')
        self._query_executor.set_prompt_context(context)

    def restore_model(self):
        self._query_executor.set_prompt_context('')


class RomeStyleModelEditor(ModelEditor):

    def __init__(self, query_executor):
        self._changed_weights = None
        super().__init__(query_executor)

    @staticmethod
    def _format_fact_for_rome(fact):
        subject = fact.get_subject_label()
        target = fact.get_target_label()
        prompt = fact.get_fact_prompt().replace(subject, '{}')
        return [{'prompt': prompt, 'subject': subject, 'target_new': {'str': target}}]

    @staticmethod
    def _format_fact_for_SWEAOS(fact, pre_fact):
        subject = fact.get_subject_label()
        target = fact.get_target_label()
        target_true = pre_fact.get_target_label()
        prompt = fact.get_fact_prompt().replace(subject, '{}')
        return [{'prompt': prompt, 'subject': subject, 'target_new': {'str': target}, 'target_true': {'str': target_true}}]

    def edit_model(self, fact):
        raise NotImplementedError()  # Override in concrete classes

    def restore_model(self):
        if self._changed_weights is None:
            return

        
        with torch.no_grad():
            for k, v in self._changed_weights.items():
                nethook.get_parameter(self._model, k)[...] = v.to(self._model_device)


class MEMITModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):


        requests = self._format_fact_for_rome(fact)
        hparams = MEMITHyperParams.from_json(f'./hparams/MEMIT/{self._model_name}.json')
        _, self._changed_weights = apply_memit_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)

class PMETModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):


        requests = self._format_fact_for_rome(fact)
        hparams = PMETHyperParams.from_json(f'./hparams/PMET/{self._model_name}.json')
        _, self._changed_weights = apply_pmet_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)



class ROMEModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):

        requests = self._format_fact_for_rome(fact)
        hparams = ROMEHyperParams.from_json(f'hparams/ROME/{self._model_name}.json')
        _, self._changed_weights = apply_rome_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)


class GRACEModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        requests = self._format_fact_for_rome(fact)
        hparams = GraceHyperParams.from_json(f'hparams/GRACE/{self._model_name}.json')
        _, self._changed_weights = apply_grace_to_model(self._model, self._tokenizer, requests, hparams, keep_original_weight=True)

    def restore_model(self):
        self._changed_weights()
        # self._model = 
class SWEAOSModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact, pre_fact):
        requests = self._format_fact_for_SWEAOS(fact, pre_fact)
        hparams = SWEAOSHyperParams.from_json(f'hparams/SWEAOS/{self._model_name}_ripple.json')
        ds_name = 'ripple'
        alg_name = 'SWEAOS'
        cache_template, optimize_cache_template = None, None
        if hparams.mode == 'kn+optimize':
            cache_template = (
                    KV_DIR
                    / f"{self._model_name.replace('/', '_')}_{alg_name}_kn_optimize"
                    / f"{ds_name}_kn_{{}}_{{}}_{{}}"
                    / f"subj_{{}}_fusion_{{}}.npz"
                )
            optimize_cache_template = (
                KV_DIR
                / f"{self._model_name.replace('/', '_')}_{alg_name}_optimize"
                / f"{ds_name}_clamp_{{}}_subj_{{}}_fusion_{{}}.npz"
            )
        elif hparams.mode == 'kn+cand':
            cache_template = (
                    KV_DIR
                    / f"{self._model_name.replace('/', '_')}_{alg_name}_kn_cand"
                    / f"{ds_name}_kn_{{}}_{{}}_{{}}"
                    / f"case_{{}}_subj_{{}}.npz"
            )
        elif hparams.mode == 'optimize':
            cache_template = (
                KV_DIR
                / f"{self._model_name.replace('/', '_')}_{alg_name}_optimize"
                / f"{ds_name}_clamp_{{}}_subj_{{}}_fusion_{{}}.npz"
            )
        else:
            raise NotImplementedError
        fused_model, _ = apply_SWEAOS_to_model(self._model, self._tokenizer, requests, hparams, cache_template=cache_template,optimize_cache_template=optimize_cache_template)
        self._model = fused_model

        
    def restore_model(self):
        self._model.unset_hook()
        self._model = self._model.model