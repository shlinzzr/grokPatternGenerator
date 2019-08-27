
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.elasticsearch.common.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.google.code.regexp.Matcher;
import com.google.code.regexp.Pattern;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class CoreNLPUtil {

	/**
	 * Logger Object 
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(CoreNLPUtil.class);


	private static CoreNLPUtil instance;
	public static Map<String, String> grokPatternDefinition;
	public static FileReader r;
	
	public static String prev_word;
	public static String prev_pos;
	public static String prev_ne;
	public static String prev_lemma;
	public static String prev_ptn;
	public static String prev_NN;
	public static Pattern GROK_PATTERN = Pattern.compile(
		      "%\\{" +
		      "(?<name>"+
		        "(?<pattern>[A-z0-9]+)"+
		          "(?::(?<subname>[A-z0-9_:]+))?"+
		          ")"+
		          "(?:=(?<definition>"+
		            "(?:"+
		            "(?:[^{}]+|\\.+)+"+
		            ")+"+
		            ")" +
		      ")?"+
		      "\\}");
	

	public static synchronized CoreNLPUtil getInstance() {
		if (instance == null) {
			instance = new CoreNLPUtil();
			
		} else {
			// do nothing 
		}

		return instance;
	}
	
	
	private CoreNLPUtil() {
		grokPatternDefinition = addPatternFromReader( new InputStreamReader(getClass().getResourceAsStream("/patterns_dir/grok.pattern")));
	}




	public String main(String text) {
		prev_word = "";      
		prev_pos = "";       
		prev_ne = "";         
		prev_lemma = "";
		prev_ptn = "";
		prev_NN = "";
		
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos");
		
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		
        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);
        
        // run all Annotators on this text
        pipeline.annotate(document);
        
        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        
        System.out.println("word\tpos\tlemma\tner");
        
        StringBuffer pattern = new StringBuffer();
        
        
        for(CoreMap sentence: sentences) {
             // traversing the words in the current sentence
             // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(TextAnnotation.class);
                // this is the POS tag of the token
                String pos = token.get(PartOfSpeechAnnotation.class);
                // this is the NER label of the token
                String ne = token.get(NamedEntityTagAnnotation.class);
                
                String lemma = token.get(LemmaAnnotation.class);
                
                
                if( token.before().length()>0 ) {
                	pattern.append(token.before());
                } 

                String ptn = patternGenerator(word, pos, ne, lemma);
                pattern.append( ptn );
                
                System.out.println(word+"\t"+pos+"\t"+ptn);
                
                prev_word = word;      
                prev_pos = pos;
                prev_ne = ne;        
                prev_lemma = lemma;   
                prev_ptn = ptn;
                
            }
        }
        
        
        String patternStr = pattern.toString();
        patternStr = patternStr.replace("%{TIMESTAMP_ISO8601}%{NUMBER}%{WORD}", "%{TIMESTAMP_ISO8601}");
        
        System.out.println("pattern=" +patternStr);
        LOGGER.info("pattern=" +patternStr);
        return patternStr;
	}
	

    public String patternGenerator(String word, String pos, String ne, String lemma) {
    	String result = "";
    	
    	
    	if (word.equals("\"") 
    	 || word.equals("``")
    	 || word.equals("''")){
    		return "\"";
    	}
    	
    	if (word.equals("'") 
    	 || word.equals("`")){
    		return "'";
    	}
    	
    	if (word.equals("-LRB-")) return "\\(";
    	if (word.equals("-RRB-")) return "\\)";
    	if (word.equals("-LSB-")) return "\\[";
    	if (word.equals("-RSB-")) return "\\]";
    	if (word.equals("-LCB-")) return "\\{";
    	if (word.equals("-RCB-")) return "\\}";
    	
    	
    	if(word.equals("<")
    	|| word.equals(">")
    	|| word.equals(",")
    	|| word.equals(":")
    	|| word.equals(";")
    	|| word.equals("=")) {
    		return word; 
    	}
    	
    	if(result.isEmpty()) result = regex(word);
    	
    	
    	/*  
    	 * "INFORMATION_SCHEMA.PROFILING" => ${WORD}
    	 *  INFORMATION =%{WORD}               
_        *  =>                         
         *  SCHEMA.PROFILING  => ${USERNAME}   
    	 */
    	if(word.equals("_") && "%{WORD}".equals(prev_ptn)) {
    		return "";
    	} else if ("_".equals(prev_word) && "%{WORD}".equals(result)) {
		    return "";
    	}
    	
    	
    	if("NN".equals(pos)
    	|| "NNP".equals(pos)
    	|| "VB".equals(pos)) {
    		prev_NN = word;
    	}
    	
    	
    	return result;
    } 
	
    
    
    public String regex(String word) {
    	String result = "%{GREEDYDATA}";
    	
		for (Map.Entry<String, String> entry : grokPatternDefinition.entrySet()) {
			try {
				Pattern pattern = this.compile(entry.getValue());
				Matcher matcher = pattern.matcher(word);
	        	if (matcher.matches()) {
	        		if("=".equals(prev_word) || ":".equals(prev_word)) {
	        			prev_NN = prev_NN.replaceAll("\\.", "_");
	        			result =  "%{" + entry.getKey() + ":" + prev_NN + "}";
	        		} else {
	        			result = "%{" + entry.getKey() + "}";
	        		}
	        		
	        		break;
	        	} 
	        	
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	
		return result;
    }
    

    public Pattern compile(String pattern) throws IOException {

    	if (pattern == null || pattern.length() == 0) {
             throw new IOException("{pattern} should not be empty or null");
         }

    	String namedRegex = pattern;
    	String originalGrokPattern = pattern;
        int index = 0;
        /** flag for infinite recurtion */
        int iteration_left = 1000;
        Boolean Continue = true;

        // Replace %{foo} with the regex (mostly groupname regex)
        // and then compile the regex
        while (Continue) {
          Continue = false;
          if (iteration_left <= 0) {
        	  throw new IOException("Deep recursion pattern compilation of " + originalGrokPattern);
          }
          iteration_left--;

          Matcher m = GROK_PATTERN.matcher(namedRegex);
          
          // Match %{Foo:bar} -> pattern name and subname
          // Match %{Foo=regex} -> add new regex definition
          if (m.find()) {
            Continue = true;
            Map<String, String> group =  m.namedGroups();
            if (group.get("definition") != null) {
              try {
                addPattern(group.get("pattern"), group.get("definition"));
                group.put("name", group.get("name") + "=" + group.get("definition"));
              } catch (IOException e) {
                // Log the exeception
              }
            }
            
            String target = "%{" + group.get("name") + "}";
            String replacement =  "(?<name" + index + ">" + grokPatternDefinition.get(group.get("pattern")) + ")";
            
            namedRegex = namedRegex.replace(target, replacement);
            
            index++;
          }
        }

        if (namedRegex.isEmpty()) {
          throw new IOException("Pattern not fount");
        }
        // Compile the regex
        return Pattern.compile(namedRegex);
      }
    
	public Map<String, String> addPatternFromReader(Reader r)  {
		grokPatternDefinition = new LinkedHashMap<String, String>(); //use linkedHashMap to setup Priortiy
		
        BufferedReader br = new BufferedReader(r);
        String line;
        // We dont want \n and commented line
        Pattern MY_PATTERN = Pattern.compile("^([A-z0-9_]+)\\s+(.*)$");
        try {
          while ((line = br.readLine()) != null) {
            Matcher m = MY_PATTERN.matcher(line);
            if (m.matches())
              addPattern(m.group(1), m.group(2));
          }
          br.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
        return grokPatternDefinition;

    }
    public void addPattern(String name, String pattern) throws IOException {
        if (name == null || pattern == null)
          throw new IOException("Invalid Pattern");
        if (name.isEmpty() || pattern.isEmpty())
          throw new IOException("Invalid Pattern");
        grokPatternDefinition.put(name, pattern);
    }
	
}
