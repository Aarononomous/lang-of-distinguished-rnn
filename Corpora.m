classdef Corpora
    properties
        corpus = cell(2,1);
        languages = {};
        allChars = [];
    end
    
    methods
        % Constructor
        function obj = Corpora(directory)
            % create the language list from filenames
            listing = dir(directory);
            c_i = 1;
            l_i = 1;
            for i = 1 : size(listing, 1)
                if (~listing(i).isdir)
                    charsUsed = '';
                    filename = listing(i).name;
                    
                    fprintf('Parsing %s\n', filename);
                    
                    splitF = split(filename, '.');
                    language = splitF(1);
                    
                    % add the new language to the list of languages
                    obj.languages{l_i} = char(language);
                    
                    % read in the file, add
                    fid = fopen(strcat(directory, '/', filename), 'r');
                    
                    tline = fgetl(fid);
                    while ischar(tline)
                        % add chars in word to charsUsed
                        charsUsed = strcat(charsUsed, tline);
                        % add [word, lang] to corpus
                        obj.corpus{1, c_i} = tline;
                        obj.corpus{2, c_i} = l_i;
                        tline = fgetl(fid);
                        c_i = c_i + 1;
                    end
                    
                    fclose(fid);
                    l_i = l_i + 1;
                    % add all chars used in this language to allChars
                    obj.allChars = strcat(obj.allChars, unique(charsUsed));
                end
            end
            
            % after parsing all files, remove dups from allChars
            obj.allChars = unique(obj.allChars);
        end
        
        function lang = decodeLang(obj, n)
            %%% returns the language at languages{n}
            if (n < 1 || n > size(obj.languages, 2))
                lang = 'unknown';
            else
                lang = obj.languages{n};
            end
        end
        
        function encoding = encodeString(obj, s)
            %%% encodes a string (i.e., char vector) s = c_1 c_2 ... into
            %%% [n_1; n_2; n_3; ...] where n_i is the index of c in allChars
            %%% if c is not in allChars, use 0
            encoding = zeros(size(s, 2), 1);
            for i = 1 : size(s, 2)
                pos = find(obj.allChars==s(i));
                if (pos)
                    encoding(i) = pos;
                else
                    encoding(i) = 0;
                end
            end
        end
    end
end