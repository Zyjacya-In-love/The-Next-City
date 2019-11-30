'''
washing 用于分词 和 借助于 stopword.txt 清洗文本
'''

import jieba
import jieba.analyse

# 创建停用词list
def stopwordlist(filepath):
    # strip() 处理的时候，如果不带参数，默认是清除两边的空白符，例如： / n, / r, / t, ' '
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordlist('stopwords20777.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords and not word.isdigit() and '.'not in word and '%' not in word and not any(char.isdigit() for char in word):
            if word != '\t':
                outstr += word
                outstr += ' '
    return outstr


# 筛选停用词
if __name__ == '__main__':
    inputs = open(r"yj_all_content\allcontent.txt", "r", encoding='utf-8')
    outputs = open(r"yj_all_content\yj_all_content_wash.txt", "w", encoding='utf-8')
    # print(inputs.tell())
    cnt = 0
    for line in inputs:
        cnt += 1
        print(cnt)
        line_seg = seg_sentence(line)
        outputs.write(line_seg + '\n')
    # print(cnt)
    outputs.close()
    inputs.close()