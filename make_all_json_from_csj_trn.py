import os
import re
from typing import List, Tuple, Dict

##
# トークナイズ用のカナ記号リストを作成
_kana_list = None
_kanas = """ア イ ウ エ オ カ キ ク ケ コ ガ ギ グ ゲ ゴ サ シ ス セ ソ
ザ ジ ズ ゼ ゾ タ チ ツ テ ト ダ デ ド ナ ニ ヌ ネ ノ ハ ヒ フ ヘ ホ
バ ビ ブ ベ ボ パ ピ プ ペ ポ マ ミ ム メ モ ラ リ ル レ ロ ヤ ユ ヨ
ワ ヲ ン ウィ ウェ ウォ キャ キュ キョ ギャ ギュ ギョ シャ シュ ショ
ジャ ジュ ジョ チャ チュ チョ ディ ドゥ デュ ニャ ニュ ニョ ヒャ ヒュ ヒョ
ビャ ビュ ビョ ピャ ピュ ピョ ミャ ミュ ミョ リャ リュ リョ イェ クヮ
グヮ シェ ジェ ティ トゥ チェ ツァ ツィ ツェ ツォ ヒェ ファ フィ フェ フォ フュ
テュ ブィ ニェ ミェ スィ ズィ ヴァ ヴィ ヴ ヴェ ヴォ ー ッ | <sp>"""
_kana_list = [x.replace(' ', '') for x in _kanas.replace('\n', ' ').split(' ')]
_kana_list = sorted(_kana_list, key=len, reverse=True)


def tokenize_kana_string(string: str) -> List[str]:
    """カナ表記の文字列を，カナ記号にトークナイズする.
    動作例:
        >>> tokenize_kana_string('エー|キョーワ<sp>ハイ|デス|ネー')
        ['エー', '|', 'キョー', 'ワ', '<sp>', 'ハイ', '|', 'デス', '|', 'ネー']
    
    Args:
        string (str): カナ表記の文字列

    Returns:
        List[str]: カナ記号のリスト
    """
    result = []
    while len(string) > 0:
        for kana in _kana_list:
            if string.startswith(kana):
                result.append(kana)
                string = string[len(kana):]
                break
        else:
            raise Exception(f'Invalid kana string: {string}')
    return result


def extract_head_tagged_string(s) -> Tuple[str, str]:
    """文字列の先頭に含まれるタグ付き文字列を抽出する.

    動作例:
        >>> extract_head_tagged_string('(D コレ(F エー)ハイ(? ア,ノ))ハイ(F エー)ハイ')
        ('(D コレ(F エー)ハイ(? ア,ノ))', 'ハイ(F エー)ハイ')

    Args:
        s (str): タグ付き文字列

    Returns:
        Tuple[str, str]: タグ付き文字列とそれ以降の文字列
    """
    if len(s) == 0:
        return '', ''
    if s[0] != '(':
        raise ValueError('not begin with (')
    stack = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            stack.pop()
            if len(stack) == 0:
                return s[:i+1], s[i+1:]
    raise ValueError('not end with )')


def split_tagged_content_with_semicolon_or_comma(content: str) -> List[str]:
    """タグが付けられた文字列を，セミコロンかカンマで分割する.
    
    動作例:
        >>> split_tagged_content_with_semicolon_or_comma('ハイ(F ウン)ハイ;ハイ')
        ['ハイ(F ウン)ハイ', 'ハイ']
        >>> split_tagged_content_with_semicolon_or_comma('ハイ(F ウン)ハイ,ハイ')
        ['ハイ(F ウン)ハイ', 'ハイ']

    Args:
        content (str): タグが付けられた文字列

    Returns:
        List[str]: 分割された文字列のリスト
    """
    result = []
    prev = 0
    open_parenthesis_count = 0
    for i, c in enumerate(content):
        if c in (';', ','):
            if open_parenthesis_count == 0:
                result.append(content[prev:i])
                prev = i + 1
        elif c == '(':
            open_parenthesis_count += 1
        elif c == ')':
            open_parenthesis_count -= 1
    result.append(content[prev:])
    return result


def tokenize_kana_string_and_extract_label_seq(string: str, tag2label: Dict[str, str]={'D2': 'D', '?': 'W'}) -> Tuple[List[str], List[str]]:
    """カナ表記の文字列（タグ付き）をカナ記号列とラベル列に変換する.

    実行例:
        >>> tokenize_kana_string_and_extract_label_seq('(D2 コレ(F エー)ハイ(? ア,ノ))ハイ(F エー)ハイ')
        ['コ', 'レ', 'エ',  'ー',  'ハ', 'イ', 'ア',  'ハ', 'イ', 'エ', 'ー', 'ハ', 'イ'], 
        ['ND', 'ND', 'NDF, 'NDF', 'D',  'D',  'ND?', 'N',  'N',  'NF', 'NF', 'N', 'N']

    説明:
        - タグを消去したカナ文字列をトークナイズする
        - トークナイズされたカナに合わせて，ラベルの列が生成される
        - ラベル
          - Nはすべてのトークンに付与される
          - タグ名がラベルとして付与される
          - タグ名とラベルが異なる場合，tag2labelで変換される
          - タグが入れ子になっている場合は複数のラベルが連結された文字列が付与される

    Args:
        string (str): カナ表記の文字列（タグ付き）
        tag2label (Dict[str, str], optional): タグ名からラベルへの変換辞書. Defaults to {'D2': 'D'}.
    
    Returns:
        Tuple[List[str], List[str]]: カナ記号列とラベル列
    """
    # 空文字列の場合はそのまま返す
    if len(string) == 0:
        return ''

    result = []
    result_label = []

    while len(string) > 0:
        match = re.match(r'^([^(]+)(.*)$', string)
        if match:
            # タグ開始以前の部分をトークナイズ
            kana_list = tokenize_kana_string(match.group(1))
            result.extend(kana_list)
            result_label.extend(['N'] * len(kana_list))

            string = match.group(2)
        else:
            # タグ
            tagged_string, string = extract_head_tagged_string(string)
            match_ = re.match(r'^\(([^ ]+) (.+)\)$', tagged_string)
            if not match_:
                if tagged_string == '(?)':
                    continue
                elif tagged_string == '(L )':
                    # 1例 (L <FV>) というものがあることを確認
                    continue
                else:
                    raise Exception('Invalid tagged string: {}'.format(tagged_string))

            tag = match_.group(1)
            content = match_.group(2)
            # タグごとにコンテンツの絞り込みをする
            if tag in ('F', 'D', 'D2', 'M', 'O', 'X', '笑', '泣', '咳', 'L'):
                # これらのタグの場合はそのまま
                pass
            else:
                # とりあえずセミコロンとカンマで分割
                contents = split_tagged_content_with_semicolon_or_comma(content)
                if tag == '?':
                    # ? の場合は最初の要素だけ
                    content = contents[0]
                elif tag in ('W', 'B'):
                    # W, B の場合は最初の要素だけ
                    content = contents[0]
                else:
                    raise Exception('Unknown tag: {}'.format(tag))

            if tag in tag2label:
                label_tag = tag2label[tag]
            else:
                label_tag = tag

            # content がタグを含む場合は再帰処理をする
            if '(' in content:
                kana_list, tag_list = tokenize_kana_string_and_extract_label_seq(content, tag2label)
                tag_list = [t + label_tag for t in tag_list]
            else:
                kana_list = tokenize_kana_string(content)
                tag_list = ['N'+label_tag] * len(kana_list)
            result.extend(kana_list)
            result_label.extend(tag_list)
                
    return result, result_label


def convert_labels_to_flags(labels, target):
    flags = []
    for label in labels:
        utt_id, label = label.split(' ', 1)
        fs = []
        for l in label.split(' '):
            if target in l:
                fs.append('1')
            else:
                fs.append('0')
        flags.append(f"{utt_id} {' '.join(fs)}")
    return flags


def make_utterance_id_from_utterance_info(lecture_id, utterance_info):
    """発話情報から発話IDを生成する.

    動作例:
        >>> make_utterance_id_from_utterance_info('A01F0001', {'start': 1.234, 'end': 5.467, 'channel': 'L'})
        'A01F0001_0001234_0005467'
        >>> make_utterance_id_from_utterance_info('D01F0001', {'start': 1.234, 'end': 5.467, 'channel': 'L'})
        'D01F0001L_0001234_0005467'
    """
    start = utterance_info['start']
    end = utterance_info['end']
    if lecture_id[0] == 'D':
        lecture_id_with_ch = lecture_id + utterance_info['channel']
    else:
        lecture_id_with_ch = lecture_id
    utt_id = f"{lecture_id_with_ch}_" + \
                f"{int(start):04d}{int(start*1000)%1000:03d}_" + \
                f"{int(end):04d}{int(end*1000)%1000:03d}"
    return utt_id    


def make_text_tokenized_kana(flattened_trn,
                             remove_word_sep=False, remove_sp=False,
                             remove_privacy_utt=True,
                             remove_extra_content=True):
    """カナ文のスペース区切りのリスト，およびパラ言語情報のスペース区切りリストを作成する

    Args:
        flattened_trn (Dict): トランスクリプトの情報
        remove_word_sep (bool, optional): 単語区切りの記号を削除するかどうか. Defaults to False.
        remove_sp (bool, optional): <sp>を削除するかどうか. Defaults to False.
        remove_privacy_utt (bool, optional): プライバシー保護のための発話を削除するかどうか. Defaults to True.
        remove_extra_content (bool, optional): 非語彙的な発話内容を削除するかどうか. Defaults to True.

    Returns:
        Tuple[List[str], List[str], List[str]]: カナ文のスペース区切りのリスト，パラ言語情報のスペース区切り，セグメント情報のスペース区切りのリスト
    """
    texts = []
    labels = []
    segments = []

    lecture_id = flattened_trn['lecture_id']
    wav_id = {'L': lecture_id}
    if lecture_id[0] == 'D':
        wav_id['L'] = lecture_id + 'L'
        wav_id['R'] = lecture_id + 'R'
    for utterance in flattened_trn['flattened_utterances']:
        utt_id = make_utterance_id_from_utterance_info(lecture_id, utterance)
        text = utterance['kana_text']

        # ボーカルフライ
        text = text.replace('<FV>', '')
        # うん/うーん/ふーん の音が特定困難な場合
        # 本来は基本形の文字列をカタカナ化したものを持ってきたい．
        text = text.replace('<VN>', 'ウン')        
        # 非語彙的な母音の引き延ばし
        text = text.replace('<H>', 'ー')
        # 非語彙的な子音の引き延ばし
        text = text.replace('<Q>', '')
        # 笑，咳，息
        text = text.replace('<笑>', '')
        text = text.replace('<咳>', '')
        text = text.replace('<息>', '')

        if remove_privacy_utt:
            if 'R' in text:
                continue
        if remove_extra_content:
            # '<' '>' で囲まれた文字列（<sp>以外）を削除
            text = re.sub(r'(<\/?(?!sp\b)[^>]+>)', '', text)
        if remove_word_sep:
            text = text.replace('|', '')
        if remove_sp:
            text = text.replace('<sp>', '')
        if len(text) == 0:
            continue

        # print(utt_id, text)
        kana_list, label_list = tokenize_kana_string_and_extract_label_seq(text)

        texts.append(f"{utt_id} {' '.join(kana_list)}")
        labels.append(f"{utt_id} {' '.join(label_list)}")
        segments.append(f"{utt_id} {wav_id[utterance['channel']]} {utterance['start']:.3f} {utterance['end']:.3f}")

    return texts, labels, segments


def remove_tag_from_kana_tagged_string(s: str):
    """タグ付き文字列からタグを除去する.
    Args:
        s (str): タグ付き文字列

    Returns:
    """

    # 空文字列の場合は～文字列を返す
    if len(s) == 0:
        return ''

    result = ''

    while len(s) > 0:
        # タグが始まる前の部分を取り出す
        match = re.match(r'^([^(]+)(.*)$', s)
        if match:
            result += match.group(1)
            s = match.group(2)
        else:
            # 最長一致でタグの部分を取り出す
            tagged_string, s = extract_head_tagged_string(s)
            match_ = re.match(r'^\(([^ ]+) (.+)\)$', tagged_string)
            if not match_:
                if tagged_string == '(?)':
                    continue
                elif tagged_string == '(L )':
                    # 1例 (L <FV>) というものがあることを確認
                    continue
                else:
                    raise Exception('Invalid tagged string: {}'.format(tagged_string))
            tag = match_.group(1)
            content = match_.group(2)
            # タグごとにコンテンツの絞り込みをする
            if tag in ('F', 'D', 'D2', 'M', 'O', 'X', '笑', '泣', '咳', 'L'):
                # これらのタグの場合はそのまま
                pass
            else:
                # とりあえずセミコロンとカンマで分割
                contents = split_tagged_content_with_semicolon_or_comma(content)
                if tag == '?':
                    # ? の場合は最初の要素だけ
                    content = contents[0]
                elif tag in ('W', 'B'):
                    # W, B の場合は最初の要素だけ
                    content = contents[0]
                else:
                    raise Exception('Unknown tag: {}'.format(tag))
            # content がタグを含む場合は再帰処理をする
            if '(' in content:
                content = remove_tag_from_kana_tagged_string(content)
            result += content
            
    return result


def make_text_kana(flattened_trn,
                   remove_word_sep=False, remove_sp=True,
                   remove_privacy_utt=True,
                   remove_extra_content=True,
                   remove_tags=True):
    """カナ文の標準テキストを作成する
    """
    texts = []
    segments = []

    lecture_id = flattened_trn['lecture_id']
    wav_id = {'L': lecture_id}
    if lecture_id[0] == 'D':
        wav_id['L'] = lecture_id + 'L'
        wav_id['R'] = lecture_id + 'R'
    for utterance in flattened_trn['flattened_utterances']:
        utt_id = make_utterance_id_from_utterance_info(lecture_id, utterance)
        text = utterance['kana_text']

        # ボーカルフライ
        text = text.replace('<FV>', '')
        # うん/うーん/ふーん の音が特定困難な場合
        # 本来は基本形の文字列をカタカナ化したものを持ってきたい．
        text = text.replace('<VN>', 'ウン')        
        # 非語彙的な母音の引き延ばし
        text = text.replace('<H>', 'ー')
        # 非語彙的な子音の引き延ばし
        text = text.replace('<Q>', '')
        # 笑，咳，息
        text = text.replace('<笑>', '')
        text = text.replace('<咳>', '')
        text = text.replace('<息>', '')

        if remove_privacy_utt:
            if 'R' in text:
                continue
        if remove_extra_content:
            # '<' '>' で囲まれた文字列を削除
            text = re.sub(r'(<\/?(?!sp\b)[^>]+>)', '', text)
        if remove_word_sep:
            text = text.replace('|', '')
        if remove_sp:
            text = text.replace('<sp>', '')
        if remove_tags:
            text = remove_tag_from_kana_tagged_string(text)
        if len(text) == 0:
            continue

        texts.append(f"{utt_id} {text}")
        segments.append(f"{utt_id} {wav_id[utterance['channel']]} {utterance['start']:.3f} {utterance['end']:.3f}")

    return texts, segments


def remove_tag_from_plain_tagged_string(s: str):
    """タグ付き文字列からタグを除去する.
    Args:
        s (str): タグ付き文字列

    Returns:
    """

    # 空文字列の場合は～文字列を返す
    if len(s) == 0:
        return ''

    result = ''

    while len(s) > 0:
        # タグが始まる前の部分を取り出す
        match = re.match(r'^([^(]+)(.*)$', s)
        if match:
            result += match.group(1)
            s = match.group(2)
        else:
            # 最長一致でタグの部分を取り出す
            tagged_string, s = extract_head_tagged_string(s)
            match_ = re.match(r'^\(([^ ]+) (.+)\)$', tagged_string)
            if not match_:
                if tagged_string == '(?)':
                    continue
                else:
                    raise Exception('Invalid tagged string: {}'.format(tagged_string))
            tag = match_.group(1)
            content = match_.group(2)
            # タグごとにコンテンツの絞り込みをする
            if tag in ('M', 'O', 'X'): # tag in ('F', 'D', 'D2', 'M', 'O', 'X',):
                pass

            elif tag == 'F':
                # F の場合はタグ<F></F>をつける
                content = "<F>" + content + "</F>"
            elif tag in ('D', 'D2'):
                # D, D2 の場合は削除
                content = ""

            else:
                # とりあえずセミコロンとカンマで分割
                contents = split_tagged_content_with_semicolon_or_comma(content)
                if tag in ('?', 'K'):
                    # ? の場合は最初の要素だけ
                    content = contents[0]
                elif tag == 'A':
                    if re.match(r'^[０-９．]+$', contents[1]):
                        content = contents[0]
                    else:
                        content = contents[1]
                else:
                    raise Exception('Unknown tag: {}'.format(tag))                    
            # content がタグを含む場合は再帰処理をする
            if '(' in content:
                content = remove_tag_from_plain_tagged_string(content)
            result += content
            
    return result


def make_text_plain(flattened_trn,
                    remove_word_sep=True, remove_sp=False,
                    remove_privacy_utt=True,
                    remove_extra_content=True,
                    remove_tags=True):
    texts = []
    segments = []

    lecture_id = flattened_trn['lecture_id']
    wav_id = {'L': lecture_id}
    if lecture_id[0] == 'D':
        wav_id['L'] = lecture_id + 'L'
        wav_id['R'] = lecture_id + 'R'
    for utterance in flattened_trn['flattened_utterances']:
        utt_id = make_utterance_id_from_utterance_info(lecture_id, utterance)
        
        text = utterance['plain_text']

        if remove_privacy_utt:
            if 'R' in text:
                continue
        if remove_extra_content:
            # <sp>以外の<>で囲まれた文字列を削除
            text = re.sub(r'(<\/?(?!sp\b)[^>]+>)', '', text)
        if remove_word_sep:
            text = text.replace('|', '')
        if remove_sp:
            text = text.replace('<sp>', '')
        if remove_tags:
            text = remove_tag_from_plain_tagged_string(text)
        if len(text) == 0:
            continue

        texts.append(f"{utt_id} {text}")
        segments.append(f"{utt_id} {wav_id[utterance['channel']]} {utterance['start']:.3f} {utterance['end']:.3f}")

    return texts, segments


def flat_utterances(parsed_or_connected_trn, 
                    word_sep='|', remove_extra_content=False, 
                    utt_sep='<sp>', ):
    """TRNファイルの解析結果の中の発話（utterance）を, 平坦化した文字列に変換する.

    Args:
        parsed_or_connected_trn (dict): parse_trn_file() または connect_utterances() の出力
        word_sep (str, optional): 単語間の区切り文字. Defaults to '|'.
        remove_extra_content (bool, optional): Trueの場合, extra_contentを除去する. Defaults to False.
        utt_sep (str, optional): 発話間の区切り文字. Defaults to '<sp>'.

    Returns:
        dict: 平坦化された発話を含む辞書型のデータ            
    """
    output = {
        'lecture_id': parsed_or_connected_trn['lecture_id'],
    }
    is_connected = 'connected_utterances' in parsed_or_connected_trn
    if is_connected:
        utterances = parsed_or_connected_trn['connected_utterances']
    else:
        utterances = parsed_or_connected_trn['utterances']
    
    flattened_utterances = []
    for utterance in utterances:
        if not is_connected:
            utterance = [utterance]
        
        if remove_extra_content:
            utterance = [u for u in utterance if 'extra_content' not in u]
        if len(utterance) == 0:
            continue

        flattend_utterance = {
            'id': utterance[0]['id'],
            'start': utterance[0]['start'],
            'end': utterance[-1]['end'],
            'channel': utterance[0]['channel'],
        }

        plain_text = ''
        kana_text = ''
        for u in utterance:
            if 'extra_content' in u:
                plain_text += u['extra_content']
                kana_text += u['extra_content']
                # print(kana_text)
            else:
                plain_text += word_sep.join([word[0] for word in u['words']])
                kana_text += word_sep.join([word[1] for word in u['words']])
            plain_text += utt_sep
            kana_text += utt_sep
        plain_text = plain_text[:-len(utt_sep)]
        kana_text = kana_text[:-len(utt_sep)]
        
        flattend_utterance['plain_text'] = plain_text
        flattend_utterance['kana_text'] = kana_text
        flattened_utterances.append(flattend_utterance)

    output['flattened_utterances'] = flattened_utterances

    return output


def connect_utterances(parsed_trn, 
                       gap: float=0.5, max: float=10.0,
                       remove_extra_content: bool=True,
                       isolate_extra_content: bool=True):
    """TRNファイルの解析結果の中の発話（utterance）を, 
    発話間のポーズや発話の長さに基づいて接続する.

    Args:
        parsed_trn (dict): parse_trn_file() の出力
        gap (float, optional): 発話間のポーズの最大長さ. この長さを超えるポーズでは発話を分割する.
                                Defaults to 0.5.
        max (float, optional): 発話の最大長さ. 接続後の発話長がこの長さを超えないようにする. Defaults to 10.0.
        remove_extra_content (bool, optional): Trueの場合, extra_contentを除去する. Defaults to True.
        isolate_extra_content (bool, optional): Trueの場合, extra_contentは単独発話として扱う. Defaults to True.
                                                remove_extra_content=True の場合は無視される.

    
    Returns:
        dict: 接続された発話を含む辞書型のデータ
    """
    output = {
        'lecture_id': parsed_trn['lecture_id'],
    }

    # utterance のリストのリスト
    connected_utterances = []
    # utterance のリスト（接続された発話のリスト）
    connected_utterance = []
    # 接続された発話の中に含まれる開いた括弧の数
    num_open_parentheses = 0
    for utterance in parsed_trn['utterances']:
        if remove_extra_content and 'extra_content' in utterance:
            continue
        local_num_open_parentheses = sum([w[0].count('(') - w[0].count(')') for w in utterance['words']])
        # print(num_open_parentheses, local_num_open_parentheses, ''.join([w[0] for w in utterance['words']]), utterance.get('extra_content', ''))
        if len(connected_utterance) == 0:
            connected_utterance.append(utterance)
            num_open_parentheses = local_num_open_parentheses
            continue
        prev_utterance = connected_utterance[-1]
        if num_open_parentheses > 0:
            if 'extra_content' not in utterance:
                connected_utterance.append(utterance)
                num_open_parentheses += local_num_open_parentheses
            else:
                # タグ内の単独の<雑音>などは接続しない
                pass
        elif isolate_extra_content and 'extra_content' in prev_utterance:
            connected_utterances.append(connected_utterance)
            connected_utterance = [utterance]
            num_open_parentheses = local_num_open_parentheses
        elif utterance['start'] - prev_utterance['end'] < gap and \
              utterance['end'] - connected_utterance[0]['start'] < max and \
              utterance['channel'] == prev_utterance['channel'] and \
              (not isolate_extra_content or 'extra_content' not in utterance):
            connected_utterance.append(utterance)
            num_open_parentheses += local_num_open_parentheses
        else:
            connected_utterances.append(connected_utterance)
            connected_utterance = [utterance]
            num_open_parentheses = local_num_open_parentheses
    if len(connected_utterance) > 0:
        connected_utterances.append(connected_utterance)
    
    output['connected_utterances'] = connected_utterances

    return output


def parse_trn_file(filename, encoding='shift_jis'):
    """TRNファイルを解析して辞書型のデータに変換する
    
    Args:
        filename (str): TRNファイルのパス
        encoding (str, optional): TRNファイルのエンコーディング. Defaults to 'shift_jis'.

    Returns:
        dict: TRNファイルの内容を辞書型に変換したもの
    """
    with open(filename, 'r', encoding=encoding) as file:
        lines = file.readlines()

    lecture_id = None
    utterances = []
    current_utterance = None

    for line in lines:
        line = line.strip()

        if line.startswith('%講演ID:'):
            lecture_id = line.split(':')[1]
        elif len(line) > 0 and line[0].isdigit():
            if current_utterance:
                utterances.append(current_utterance)
            utterance_id, time_range, channel = line.split(' ')
            channel, extra_content = channel.split(':')
            start_time, end_time = map(float, time_range.split('-'))
            current_utterance = {
                'id': int(utterance_id),
                'start': start_time,
                'end': end_time,
                'channel': channel,
                'words': []
            }
            if extra_content != '':
                current_utterance['extra_content'] = extra_content
        elif line.startswith('%'):
            continue
        else:
            words = line.split('&')
            assert len(words) == 2
            words = tuple([word.strip() for word in words])
            current_utterance['words'].append(words)

    if current_utterance:
        utterances.append(current_utterance)

    output = {
        'lecture_id': lecture_id,
        'utterances': utterances
    }

    return output

def make_wav_scp(wav_file_path):
    basename = os.path.basename(wav_file_path)
    lecture_id = basename[:basename.find('.')]
    if lecture_id[0] != 'D':
        return ['{} {}'.format(lecture_id, wav_file_path)]
    else:
        return ['{} sox {} -t wav - remix 1|'.format(lecture_id+'L', wav_file_path),
                '{} sox {} -t wav - remix 2|'.format(lecture_id+'R', wav_file_path)]
    
def make_utt2spk(texts):
    result = []
    for text in texts:
        # print(text)
        utt_id, _ = text.split(' ', 1)
        spk_id = utt_id[:utt_id.find('_')]
        result.append('{} {}'.format(utt_id, spk_id))
    return result

def sort_texts(texts):
    texts_dict = {k: v for k, v in [text.split(' ', 1) for text in texts]}
    sorted_keys = sorted(texts_dict.keys())
    return [k + ' ' + texts_dict[k] for k in sorted_keys]

def sort_as_texts(data, texts):
    data_dict = {k: v for k, v in [d.split(' ', 1) for d in data]}
    return [k + ' ' + data_dict[k] for k in [t.split(' ', 1)[0] for t in texts]]

def combine_texts_with_labels(texts, labels, para_info):
    para_keys = list(para_info.keys())
    labels = sort_as_texts(labels, texts)
    
    result = []
    for t, l in zip(texts, labels):
        utt_id_text, tbody = t.split(' ', 1)
        utt_id_label, lbody = l.split(' ', 1)
        assert utt_id_text == utt_id_label
        result_text = []
        for t_token, l_token in zip(tbody.split(' '), lbody.split(' ')):
            for para_key in para_keys:
                if para_key in l_token:
                    t_token = t_token + f"+{para_key}"
            result_text.append(t_token)
        result.append(utt_id_text + ' ' + ' '.join(result_text))
    return result
    

def main(trn_file_path, encoding='shift_jis', gap=0.5, max=10.0,):
    """TRNファイルを読み込んで, テキストファイルを出力する

    Args:
        trn_file_path (str): TRNファイルのパス
        encoding (str, optional): TRNファイルのエンコーディング. Defaults to 'shift_jis'.
        gap (float, optional): 発話を接続する際の最大の間隔. Defaults to 0.5.
        max (float, optional): 1つの発話に含める最大の時間. Defaults to 10.0.
    """
    parsed = parse_trn_file(trn_file_path, encoding=encoding)
    connected = connect_utterances(parsed, gap=gap, max=max, remove_extra_content=True)
    flattened = flat_utterances(connected)

    texts, _ = make_text_plain(flattened,
                               remove_word_sep=True,
                               remove_sp=False,
                               remove_privacy_utt=True,
                               remove_extra_content=True,
                               remove_tags=True)
    plain_texts = texts
    
    texts, labels, _ = make_text_tokenized_kana(flattened)
    texts = combine_texts_with_labels(texts, labels, {'F': 'Filler', 'D': 'Disfluency'})
    tokenized_kana_texts = texts

    plain_texts_dict = {k: v for k, v in [text.split(' ', 1) for text in plain_texts]}
    tokenized_kana_texts_dict = {k: v for k, v in [text.split(' ', 1) for text in tokenized_kana_texts]}

    results = []
    for k in plain_texts_dict.keys():
        if k not in tokenized_kana_texts_dict:
            print(f"Warning: {k} is not found in tokenized kana texts")
            continue
        results.append({'utt_id': k,
                        'plain_text': plain_texts_dict[k],
                        'tokenized_kana_text': tokenized_kana_texts_dict[k]})
    return results


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSJ TRN files to json format')
    parser.add_argument('trn_top_dir', type=str, help='CSJ TRN top directory')
    parser.add_argument('--encoding', type=str, default='shift_jis', help='TRN file encoding')
    parser.add_argument('--gap', type=float, default=0.5, help='Maximum gap between utterances')
    parser.add_argument('--max', type=float, default=10.0, help='Maximum duration for one utterance')
    args = parser.parse_args()

    import glob
    import os
    from tqdm import tqdm

    result = []
    generator = tqdm(sorted(glob.glob(os.path.join(args.trn_top_dir, '**', '*.trn'), recursive=True)))
    for trn_file_path in generator:
        result.extend(main(trn_file_path, encoding=args.encoding, gap=args.gap, max=args.max, ))
        generator.set_postfix({'num': len(result)})
    import json
    json.dump(result, open('all.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

