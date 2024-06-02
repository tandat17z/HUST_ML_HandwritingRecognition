import Levenshtein

class EvalMetrics:
    def __init__(self) -> None:
        self.cer = 0
        self.wer = 0
        self.num = 0

    def add(self, preds, labels):
        self.num += len(preds)
        for i in range(preds.__len__()):
            sent_pred = preds[i]
            sent_label = labels[i]
            c_cost = Levenshtein.distance(sent_pred, sent_label)
            w_cost = Levenshtein.distance(sent_pred.split(), sent_label.split())

            self.cer += c_cost/len(sent_pred) if len(sent_pred) != 0 else 1
            self.wer += w_cost/len(sent_pred.split()) if len(sent_pred.split()) != 0 else 1

    def eval(self):
        return self.cer/self.num, self.wer/self.num