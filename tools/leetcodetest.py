def isValid( s):
    length = len(s)
    for i in range(length):
        j = i + 1
        while s:
            if s[i] == '(' and s[j] == ')':
                s = s.replace(s[i], '')
                s = s.replace(s[j - 1], '')
            elif s[i] == '[' and s[j] == ']':
                s = s.replace(s[i], '')
                s = s.replace(s[j - 1], '')
            elif s[i] == '{' and s[j] == '}':
                s = s.replace(s[i], '')
                s = s.replace(s[j - 1], '')

    if s == "":

        return 'true'
    else:
        return 'false'
if __name__ == '__main__':
    s = "()"

    print(isValid(s))