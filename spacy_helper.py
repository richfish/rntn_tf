# if you want to sub NER or annotate input
# from wingorad schema project

import spacy
from spacy.en import English
from spacy import attrs
parser = English()

import pdb

class SpacyHelper():

    def __init__(self):
        self.parser = English()

    thuman = parser(u"human")
    tperson = parser(u"person")
    tevent = parser(u"event")
    tthing = parser(u"thing")
    tplace = parser(u"place")
    sim_test_items = [thuman, tperson, tevent, tthing, tplace]

    ner_labels = { #ref with self.ner_labels
        "GPE": "[location]",
        "PERSON": "[person]",
        "QUANTITY": "[quantity]",
        "MONEY": "[money]",
        "ORG": "[organization]",
        "DATE": "[date]",
        "TIME": "[time]",
        "PERCENT": "[percent]",
        "PRODUCT": "[product]",
        "NORP": "[group]",
        "FAC": "[facility]",
        "LAW": "[law]",
        "LANGUAGE": "[language]",
        "WORK_OF_ART": "[work-of-art]",
        "LOC": "[location]",
        "EVENT": "[event]",
        "": ""
    }

    def parse(self, sent):
        return self.parser(sent)

    def determine_main_head_type(self, sent):
        sent_len = len(sent)
        head_word = None
        try: # can wide range of errors for weird phrases "e.g. (used as intensive) every"
            for i, token in enumerate(sent):
                if len([x for x in token.subtree]) == sent_len: #subtree may not be best approach...
                    head_word = sent[i]
                    break
            return [head_word, head_word.pos_]
        except:
            return False

    def get_nsubj_phrase(self, sent):
        return [x for x in self.get_nsubj_word(sent).subtree]

    def get_nsubj_word(self, sent):
        for x in sent:
            if x.dep_ == "nsubj":
                return x

    def get_all_np(self,sent,re_parse=True):
        #cant be Span, must be doc
        if re_parse:
            sent = self.parse(unicode(sent.text))
        return list(sent.noun_chunks)

    def get_all_vps(self, sent):
        vps = []
        for se in sent:
            if se.pos_ == "VERB":
                vps.append([x for x in se.subtree])
        return vps

    def get_all_verbs(self, sent):
        return [x for x in sent if x.pos_ == "VERB"]

    # def get_vphrase_chunks(self, sent):
    #     v_chunks = []
    #     for n_chunk in sent.noun_chunks:
    #         subtree = [x for x in v_chunk.subtree]
    #         if len(subtree) > 2:
    #             v_chunks.append(subtree)
    #     return v_chunks

    # be wary of breaking up the word1-word2 phrases...
    def get_nphrase_chunks(self, sent):
        # crude heuristic, yes
        n_chunks = []
        for n_chunk in sent.noun_chunks:
            subtree = [x for x in n_chunk.subtree]
            if len(subtree) > 2:
                n_chunks.append(subtree)
        return n_chunks

    def get_nphrase_chunks_all(self, sent):
        # crude heuristic, yes
        n_chunks = []
        n_chunks_final = []
        for n_chunk in sent.noun_chunks:
            subtree = [x for x in n_chunk.subtree]
            if len(subtree) > 0:
                n_chunks.append(subtree)
        for i, chunk in enumerate(n_chunks):  #need to filter out dups
            c_len = len(chunk)
            if len(filter(lambda x: x in n_chunks[i-1], chunk)) == c_len:
                continue
            n_chunks_final.append(chunk)
        return n_chunks_final


    def get_all_dep_labels(self, sent):
        return [x.dep_ for x in sent]

    def similarity_score_type(self, word):
        scores = []
        for test in self.sim_test_items:
            scores.append(word.similarity(test))
        max_score = max(scores)
        closest_type = self.sim_test_items[scores.index(max_score)][0].lemma_
        return closest_type

    def is_a_person(self, word):
        # high threshold...
        # not good for is-a analysis I don't think...
        pass

    def get_proper_sents(self, sents):
        propers = []
        for sent in sents:
            if self.is_proper_sentence(sent):
                propers.append(sent)
        return propers

    def get_partial_phrases(self, sents):
        partials = []
        for sent in sents:
            if not self.is_proper_sentence(sent):
                partials.append(sent)
        #maybe remove Quotes like best grief is tongueless"- Emily Dickinson, or handle later
        return partials

    def is_proper_sentence(self, sent):
        head_type = self.determine_main_head_type(sent)
        if head_type:
            return (head_type[1] == "VERB") and ('nsubj' in self.get_all_dep_labels(sent))
        else:
            return False

    def substitute_ner_labels(self,sent):
        # Note, NER ents are somtimes more than 1 word...
        skip_ents = ['ORDINAL', 'CARDINAL']
        after_sub = []
        entities = [x.text for x in sent.ents]
        skip_is = []
        for i, token in enumerate(sent):
            if i in skip_is:
                continue
            if token.ent_type_ in skip_ents:
                after_sub.append(token)
                continue
            if token.text in entities:
                label = self.ner_labels[token.ent_type_]
                after_sub.append(label)
            elif (i+1 < len(sent) and (token.text + " " + sent[i+1].text) in entities):
                label1 = self.ner_labels[token.ent_type_]
                if sent[i+1].ent_type_ in skip_ents:
                    label2 = token
                else:
                    label2 = self.ner_labels[sent[i+1].ent_type_]
                if label1 == label2:
                    after_sub.append(label1)
                    skip_is.append(i+1)
                else:
                    after_sub.append(label1)
                    after_sub.append(label2)
            else:
                after_sub.append(token)
        return after_sub
