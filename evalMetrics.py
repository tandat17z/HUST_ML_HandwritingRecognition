import Levenshtein

class EvalMetrics:
    def __init__(self) -> None:
        self.cer = 0
        self.wer = 0
        self.num = 0

    def add(self, preds, labels):
        assert len(preds) == len(labels), "error in Eval Metrics"
        self.num += len(preds)
        for i in range(preds.__len__()):
            sent_pred = preds[i]
            sent_label = labels[i]
            c_cost = Levenshtein.distance(sent_pred, sent_label)
            w_cost = Levenshtein.distance(sent_pred.split(), sent_label.split())

            l1 = max(len(sent_pred), len(sent_label))
            l2 = max(len(sent_pred.split()), len(sent_label.split()))
            self.cer += c_cost/l1 if l1 != 0 else 1
            self.wer += w_cost/l2 if l2 != 0 else 1
            # print(self.cer, self.wer)

    def eval(self):
        return self.cer/self.num, self.wer/self.num