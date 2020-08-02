import typing


class Label:

    def __init__(self, value: int, label: str, color: list):
        self.value = value
        self.label = label
        self.color = color

    def __str__(self):
        return f'{self.value}, {self.label}'


class ITKSNAPLabelFileParser:

    @staticmethod
    def parse(file: str) -> typing.List[Label]:
        labels = []
        with open(file, 'r') as f:
            for line in f:
                # lines starting with # contain comments
                if not line.startswith('#'):
                    labels.append(ITKSNAPLabelFileParser.parse_line(line))

        return labels

    @staticmethod
    def parse_line(line: str):
        data = line.split()
        return Label(int(data[0]), data[-1].replace('"', ''), [data[1], data[2], data[3]])


def get_thigh_label_values():
    return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37]


def get_leg_label_values():
    return [8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35]


def filter_labels_by_values(labels: typing.List[Label], ids: typing.List[int]):
    return [lbl for lbl in labels if lbl.value in ids]
