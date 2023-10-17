package au.edu.uq.imb.memesuite.updatedb;
import au.edu.uq.imb.memesuite.data.AlphStd;
import au.edu.uq.imb.memesuite.util.MultiSourceStatus;

import org.apache.commons.net.PrintCommandListener;
import org.apache.commons.net.ftp.FTPClient;
import org.apache.commons.net.ftp.FTPFile;
import org.sqlite.SQLiteDataSource;

import java.io.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.*;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.JSONValue;
import org.json.simple.parser.ParseException;
import org.json.simple.parser.JSONParser;

/**
 * This updates the given Ensembl database division.
 */
public class EnsemblUpdater extends SequenceUpdater {
  private String rest_host;
  private String seq_host;
  private String ftpLog;
  private int retrieverId;
  private String Division;
  private boolean isBacteria;
  private boolean isVertebrates;
  private String DbRegexp;
  private Integer DbRel;
  private boolean abinitio;
  private String categoryName;
  private String restSubdir;
  private String dbSubdir;
  private String ftpSubdir;
  private Logger logger;

  private Map<String, String> name2Subdir;

  private static final Pattern NUM_RE = Pattern.compile("^release-(\\d+)$");
  private static final Pattern ABINITIO_RE = Pattern.compile("^.*abinitio$");

  public EnsemblUpdater(String division, SQLiteDataSource dataSource,
      ReentrantReadWriteLock dbLock, File binDir, File dbDir, String dbRegexp, String dbRel,
      ExecutorService worker, MultiSourceStatus statusManager) {
    super("Ensembl Updater", dataSource, dbLock, binDir, dbDir, worker, statusManager);
    Division = division;
    isBacteria = Division.equals("ensemblbacteria");
    isVertebrates = (Division.equals("ensemblvertebrates") || Division.equals("ensemblvertebratesabinitio"));
    DbRegexp = dbRegexp;
    try {
      DbRel = Integer.parseInt(dbRel);
    } catch(NumberFormatException e) {
      DbRel = -1;			// Just list the releases available.
    }

    // Get the common properties.
    Properties conf = loadConf(EnsemblUpdater.class, dbDir, "EnsemblUpdater.properties");
    Matcher m = ABINITIO_RE.matcher(Division);
    abinitio = m.matches();
    rest_host = conf.getProperty("rest_host", "https://rest.ensembl.org").trim();
    if (isVertebrates) {
      seq_host = conf.getProperty("vert_host", "ftp.ensembl.org").trim();
    } else {
      seq_host = conf.getProperty("other_host", "ftp.ensemblgenomes.org").trim();
    }
    ftpLog = conf.getProperty("ftp.log", "").trim();
    // Get the properties specific to this division.
    retrieverId = Integer.valueOf(conf.getProperty(division + "_retrieverId", "3").trim());
    categoryName = conf.getProperty(division + "_categoryName", "unknown_categoryName").trim();
    restSubdir = conf.getProperty(division + "_restSubdir", "unknown_restSubdir").trim();
    dbSubdir = restSubdir + (abinitio ? "Abinitio" : "");
    ftpSubdir = conf.getProperty(division + "_ftpSubdir", "unknown_ftpSubdir").trim();
    logger = Logger.getLogger("au.edu.uq.imb.memesuite.updatedb.ensembl." + ftpSubdir);
    logger.log(Level.INFO, "Ensembl division: " + division);
  }

  @Override
  public Void call() {
    int errorCount = 0;
    FTPClient seq_ftp = null;
    File dbTarget;
    int genomeRelease = 0;
    String releaseType = isVertebrates ? "Ensembl Release" : "Ensembl Genomes Release";

    progress.setTask("Starting");


    // Vertebrates use "Ensembl release" numbers, others use "Ensembl Genomes" release numbers.
    // File names for Bacteria changed after Ensembl Genomes Release 28.
    if (isBacteria && (DbRel > 0 && DbRel <= 28)) {
      System.out.println("\nERROR: Only Ensembl Genomes releases after release 28 are currently supported for Bacteria.");
      logger.log(Level.SEVERE, "Only Ensembl Genomes releases after release 28 are currently supported for Bacteria.");
      return null;
    }
    // The directory structure changed after Ensembl Release 46, and we don't support earlier.
    if (isVertebrates && (DbRel > 0 && DbRel <= 46)) {
      System.out.println("\nERROR: Only Ensembl releases after release 46 are currently supported for Vertebrates.");
      logger.log(Level.SEVERE, "Only Ensembl releases after release 46 are currently supported for Vertebrates.");
      return null;
    }

    try {
      logger.log(Level.INFO, "categoryName: " + categoryName);

      // Create subdirectory.
      dbTarget = new File(dbDir + "/" + dbSubdir);
      logger.log(Level.INFO, "Creating subdirectory for genomes and proteins: " + dbTarget);
      if (!dbTarget.exists() && !dbTarget.mkdir()) {
        logger.log(Level.SEVERE, "Unable to create subdirectory " + dbTarget + "!");
        return null;
      }

      // Login to Ensembl sequences FTP site.
      seq_ftp = new FTPClient();
      if (!ftpLog.isEmpty()) {
	seq_ftp.addProtocolCommandListener(new PrintCommandListener(new PrintWriter(new FileOutputStream(ftpLog)), true));
      }
      if (!loginFtp(seq_ftp, seq_host)) return null;

      //
      // Get information on available releases.
      //
      if (DbRel == -1) {
        System.out.println("\nEnsembl Updater - Listing Available Releases: ");
      } else if (DbRel == 0) {
	System.out.println("\nEnsembl Updater - Getting current " + releaseType + " number");
      } else {
	System.out.println("");
      }
      // Get the minimum and maximum release number.
      FTPFile[] directories = seq_ftp.listDirectories("/pub");
      Integer minRel=1000000, maxRel=0;
      for (FTPFile dir : directories) {
        Matcher m = NUM_RE.matcher(dir.getName());
        if (!m.matches()) continue;
        int rel = Integer.parseInt(m.group(1), 10);
        if (rel < minRel) {
          minRel = rel;
        } else if (rel > maxRel) {
          maxRel = rel;
        }
      }
      Integer[] releaseNumbers = new Integer[directories.length]; 
      int nreleases = 0;
      if (DbRel == 0) {			// Use current release.
        genomeRelease = maxRel;		
      } else if (DbRel > 0) {		// Use specified release.
        if (DbRel >= minRel && DbRel <= maxRel) {
          if (isVertebrates) {
            genomeRelease = DbRel;
          } else {
            FTPFile[] subdirs = seq_ftp.listDirectories("/pub/release-" + DbRel + "/" + ftpSubdir);
            if (subdirs.length > 0) {
              genomeRelease = DbRel;
            } else {
              System.out.println("ERROR: " + releaseType + " " + DbRel + " does not exist!");
              return null;
            }
          }
        }
      } else {				// List releases.
        int rel, validRel=0;
        for (rel=minRel; rel<=maxRel; rel++) {
          if (isVertebrates || validRel > 0 ) {
            releaseNumbers[nreleases++] = rel;
          } else {			// Check if release is valid for this category.
            FTPFile[] subdirs = seq_ftp.listDirectories("/pub/release-" + rel + "/" + ftpSubdir);
            if (subdirs.length > 0) {
              validRel = rel;
              releaseNumbers[nreleases++] = rel;
            }
          }
        }
      }

      if (DbRel >= 0) System.out.println("Ensembl Updater - Using " + releaseType + " number " + genomeRelease);
      // Check that the desired release was found (or list releases).
      if (DbRel == -1) {
        Integer[] resizedArray = Arrays.copyOf(releaseNumbers, nreleases);
        Arrays.sort(resizedArray);
        for (int rel : resizedArray) {
          System.out.println(Division + " " + releaseType + ": " + rel);
        }
        return null;
      } else if (genomeRelease == 0) {
        System.out.println("ERROR: Failed finding desired " + releaseType + "!");
        logger.log(Level.SEVERE, "Failed finding desired " + releaseType + "!");
        return null;
      }

      // Get paths to genomes in this Ensembl division.
      System.out.println("Ensembl Updater - Getting paths to genomes in division '" + Division + "' for " + releaseType + " " + genomeRelease);
      String rootDir;
      if (isVertebrates) {
        rootDir = "/pub/release-" + genomeRelease + "/fasta/";
      } else {
	rootDir = "/pub/release-" + genomeRelease + "/" + ftpSubdir + "/fasta/";
      }
      name2Subdir = makeMapName2Subdir(seq_ftp, rootDir);
      Set set=name2Subdir.entrySet();	// Converting to Set so that we can traverse.
      Iterator itr=set.iterator();  
      while (itr.hasNext()) {
	// Converting to Map.Entry so that we can get key and value separately.
	Map.Entry entry=(Map.Entry)itr.next();  
      }

      // Get the list of available genomes from the REST server, filtered by available directories.
      List<EnsemblGenome> genomes = ensemblAvailableGenomes(String.valueOf(genomeRelease), abinitio, retrieverId, categoryName, rest_host, seq_host, rootDir, restSubdir, name2Subdir);
      if (genomes.isEmpty()) {
        logger.log(Level.SEVERE, "Failed parsing the finding genomes using the Ensembl REST interface... No genomes found!");
        return null;
      }

      // Query the database to see which genomes we already have
      // and remove them so we don't download them again.  
      // Only download genomes whose genome name matches the species regexp.
      int navailable_genomes = 0;
      int nhave_genomes = 0;
      int nskipped_genomes = 0;
      int nmatched_genomes = 0;
      int nmatched_missing = 0;
      Pattern db_regexp = Pattern.compile(DbRegexp, Pattern.CASE_INSENSITIVE);
      Iterator<EnsemblGenome> iterator = genomes.iterator();
      String[] missingGenomes = new String[genomes.size()];
      String[] alreadyHaveGenomes = new String[genomes.size()];
      int nmissing = 0;
      int nalreadyHave= 0;
      while (iterator.hasNext()) {
        EnsemblGenome genome = iterator.next();
        navailable_genomes++;
        String listing_name = String.valueOf(genome);
        Matcher matcher = db_regexp.matcher(listing_name);
        boolean matchFound = matcher.find();
        if (matchFound) nmatched_genomes++;
        // Remove from genomes list the genomes that we already have.
        if (sourceExists(genome, true)) {
          iterator.remove();
          // Warn about genomes that match the regular expression that we already have
	  // unless we are downloading or listing everything.
	  if (matchFound && ! DbRegexp.equals(".*") && ! DbRegexp.equals("^$")) {
            logger.log(Level.INFO, "Already have " + genome);
	    alreadyHaveGenomes[nalreadyHave++] = "Already have: " + listing_name;
          }
          nhave_genomes++;
        } else {
          if (matchFound) nmatched_missing++;
	  // Remove from genomes list the genomes that don't match the regular expression.
	  if (! matchFound) {
	    if (DbRegexp.equals("^$")) missingGenomes[nmissing++] = "missing: " + listing_name;
	    iterator.remove();
	    nskipped_genomes++;
	  }
        }
        System.out.print("\rEnsembl Updater - FASTA files: " + navailable_genomes + "; Already have: " + nhave_genomes);
        if (! DbRegexp.equals("^$")) System.out.print("; Matching regexp: " + nmatched_genomes + "; Matching but missing: " + nmatched_missing);
      }
      System.out.println("");
      // Print number of matching files.
      if (! DbRegexp.equals("^$")) {
	System.out.println("Ensembl Updater - " + nmatched_genomes + '/' + navailable_genomes + " available FASTA files match the species regular expression '" + DbRegexp + "'");
	System.out.println("Ensembl Updater - " + nmatched_missing + '/' + nmatched_genomes + " missing FASTA files match the species regular expression '" + DbRegexp + "'");
      } else {
	System.out.println("Ensembl Updater - Missing " + nmissing + " out of " + navailable_genomes + " available FASTA files");
      }
      // Print already have genomes, sorted.
      if (nalreadyHave > 0) {
        System.out.println("Ensembl Updater - Listing matching FASTA files we already have from Division '" + Division + "':\n");
        String[] resizedArray = Arrays.copyOf(alreadyHaveGenomes, nalreadyHave);
        Arrays.sort(resizedArray);
        for (String alreadyHave : resizedArray) {
          System.out.println(alreadyHave);
        }
      }
      // Print missing genomes, sorted.
      if (DbRegexp.equals("^$") && nmissing > 0) {
        System.out.println("Ensembl Updater - Listing Missing FASTA files from Division '" + Division + "':\n");
        String[] resizedArray = Arrays.copyOf(missingGenomes, nmissing);
        Arrays.sort(resizedArray);
        for (String missing : resizedArray) {
          System.out.println(missing);
        }
      }
      logger.log(Level.INFO, "Number of genomes AVAILABLE: " + navailable_genomes);
      logger.log(Level.INFO, "Number of genomes HAVE: " + nhave_genomes);
      logger.log(Level.INFO, "Number of genomes SKIPPED: " + nskipped_genomes);
      logger.log(Level.INFO, "Number of genomes to UPDATE: " + (navailable_genomes - nskipped_genomes - nhave_genomes));

      if (DbRegexp != "^$") System.out.println("Ensembl Updater - Downloading " + genomes.size() + " missing FASTA files");
      for (EnsemblGenome genome : genomes) {
        try {
          checkWorkerTasks();
          if (!determineFtpSource(seq_host, seq_ftp, genome, RETRY_COUNT)) continue;
          checkWorkerTasks();
          if (!downloadFtpSource(seq_ftp, genome, true, RETRY_COUNT)) continue;
	  System.out.print("\rEnsembl Updater - Downloading FASTA file: '" + genome + "'");
          enqueueSequences(new EnsemblSequenceProcessor(genome, dbTarget));
          nhave_genomes++;
        } catch (IOException e) {
          logger.log(Level.WARNING, "Skipped " + genome + " due to ftp errors", e);
          errorCount++;
          if (errorCount >= ERROR_COUNT) throw new IOException("Too many IO Exceptions", e);
        }
      }
      seq_ftp.logout();
      seq_ftp.disconnect();
      progress.complete();
      waitForWorkerTasks();
      // Report numbers of Available and Missing sequence files.
      int nmissing_genomes = navailable_genomes - nhave_genomes;
      String a_verb = (navailable_genomes == 1) ? " is " : " are ";
      String m_verb = (nmissing_genomes == 1) ? " is " : " are ";
      System.out.println("\nEnsembl Updater - There" + a_verb + navailable_genomes + 
        " available sequence file(s) and " + (navailable_genomes - nhave_genomes) + m_verb + 
        "missing for " + releaseType + " " + genomeRelease + ".\n*******");
      logger.log(Level.INFO, "Finished update of " + categoryName);
    } catch (ExecutionException e) { // only thrown by sequence processor
      cancelWorkerTasks();
      logger.log(Level.SEVERE, "Abandoning update of " + dbSubdir + " due to failure to process sequences!", e);
    } catch (SQLException e) {
      logger.log(Level.SEVERE, "Abandoning update of " + dbSubdir + "!", e);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Abandoning update of " + dbSubdir + "!", e);
    } catch (InterruptedException e) {
      logger.log(Level.WARNING, "Update of " + dbSubdir + " was interrupted!");
    } catch (RuntimeException e) {
      logger.log(Level.SEVERE, "RuntimeException!", e);
      throw e;
    } catch (Error e) {
      logger.log(Level.SEVERE, "Error!", e);
      throw e;
    } finally {
      if (seq_ftp != null && seq_ftp.isConnected()) {
        try {
          seq_ftp.logout();
        } catch (IOException e) { /* ignore */ }
        try {
          seq_ftp.disconnect();
        } catch (IOException e) { /* ignore */ }
      }
      progress.complete();
    }
    return null;
  }

  private class EnsemblSequenceProcessor extends SequenceProcessor {
    private EnsemblGenome genome;
    private File dbTarget;

    public EnsemblSequenceProcessor(EnsemblGenome genome, File dbTarget) {
      super(dataSource, dbLock, binDir, dbDir, status);
      this.genome = genome;
      this.dbTarget = dbTarget;
    }

    @Override
    public void process() throws IOException, SQLException, InterruptedException {
      if (!unpackSequences(genome, dbTarget)) return;
      processSequences(genome, dbTarget);
      recordSequences(genome, dbTarget, dbSubdir);
    }
  }

  // Read a JSON object from the Ensembl REST interface.
  protected String ensemblRestRead(URL url) {
    String output = "";
    Reader reader = null;

    try {
      URLConnection connection = url.openConnection();
      HttpURLConnection httpConnection = (HttpURLConnection)connection;

      httpConnection.setRequestProperty("Content-Type", "application/json");

      InputStream response = connection.getInputStream();
      int responseCode = httpConnection.getResponseCode();

      if (responseCode != 200) {
        throw new RuntimeException("Response code was not 200. Detected response was " + responseCode);
      }

      // Read from the REST interface.
      reader = new BufferedReader(new InputStreamReader(response, "UTF-8"));
      StringBuilder builder = new StringBuilder();
      char[] buffer = new char[8192];
      int read;
      while ((read = reader.read(buffer, 0, buffer.length)) > 0) {
        builder.append(buffer, 0, read);
      }
      output = builder.toString();
    } catch (IOException e) {
     logger.log(Level.SEVERE, "URL exception.");
    } finally {
      if (reader != null) try {
        reader.close();
      } catch (IOException logOrIgnore) {
        logOrIgnore.printStackTrace();
      }
    }

    return output;
  } // ensemblRestRead()

  //
  // Make a Map of from the name of the species to its directory.
  // The species directory MUST be either:
  //   1) in the baseDir, or
  //   2) in a subdirectory of it named "*_collection".
  //
  protected Map<String, String> makeMapName2Subdir(
    FTPClient ftpClient, String rootDir
  ) throws IOException {
    Map<String, String> map = new HashMap<>();

    FTPFile[] topFiles = ftpClient.listFiles(rootDir);
    System.out.println("Ensembl Updater - Getting list of remote genomes in " + topFiles.length + " directories");
    for (FTPFile aFile : topFiles) {
      Matcher m = collectionPattern.matcher(aFile.getName());
      if (m.find()) {
        String collectionName = aFile.getName();
        String subDir = rootDir + collectionName + "/";
        FTPFile[] subFiles = ftpClient.listFiles(subDir);
        for (FTPFile bFile : subFiles) {
	  //logger.log(Level.INFO, "Mapping " + bFile.getName() + " to " + collectionName);
          map.put(bFile.getName(), collectionName + "/");
        }
      } else {
	//logger.log(Level.INFO, "Mapping " + aFile.getName() + " to nothing ");
        map.put(aFile.getName(), "");
      }
      System.out.print("\rEnsembl Updater - Found " + map.size() + " remote genome directories");
    }
    System.out.print("\n");
    return map;
  } // makeMapName2Subdir

  // Get the list of available genomes.
  protected List<EnsemblGenome> ensemblAvailableGenomes(
    String genomeRelease, boolean abinitio, int retrieverId, String categoryName,
    String rest_host, String host, String rootDir, String restSubdir, Map<String, String> name2Subdir
  ) {
  
    List<EnsemblGenome> genomes = new ArrayList<EnsemblGenome>();

    // get a listing of species
    try {
      logger.log(Level.INFO, "Starting Update of " + categoryName);

      // Create the REST query.
      String ext = "/info/genomes/division/" + restSubdir + "?";
      URL url = new URL(rest_host + ext);

      // Query the REST interface.
      String output = ensemblRestRead(url);

      // Parse the JSON string.
      JSONParser parser = new JSONParser();
      try {
	Object obj = parser.parse(output);
	JSONArray array = (JSONArray) obj;
	Iterator itr = array.iterator();
	boolean first_genome = true;
        int ntested = 0;
	while (itr.hasNext()) {
          ntested++;
	  JSONObject jenome = (JSONObject) itr.next();
	  String displayName = jenome.get("display_name").toString();
	  logger.log(Level.INFO, "Display Name: " + displayName);
	  String scientificName = jenome.get("scientific_name").toString();
	  String ftpDirName = jenome.get("name").toString();
	  String path = rootDir;
	  // Add the collection name to the path; skip if there is no match to ftpDirName.
	  String collection = name2Subdir.get(ftpDirName);
	  if (collection == null) {
            continue;			// skip; not available
          } else {
            path += collection; 	// Add the collection name to the path.
          }
	  path += ftpDirName;
	  String dnaPath = path + "/dna/";
	  String proteinPath = path + "/pep/";
	  logger.log(Level.FINE, "Adding Ensembl DNA and protein genomes for " + displayName);
	  if (! abinitio) {
            // Download the soft-masked versions if available, otherwise, download the un-masked versions.
            // Prefer the primary_assembly, then the top_level.
	    genomes.add(new EnsemblGenome(genomeRelease, retrieverId, categoryName, displayName, ftpDirName, AlphStd.DNA, false, dnaPath, host, dnaFilePattern3, dnaFilePattern4, dnaFilePattern1, dnaFilePattern2));
	  }
	  genomes.add(new EnsemblGenome(genomeRelease, retrieverId, categoryName, displayName, ftpDirName, AlphStd.PROTEIN, abinitio, proteinPath, host, abinitio ? aiFilePattern : aaFilePattern));
          System.out.print("\rEnsembl Updater - Found " + genomes.size() + " available FASTA files");
	}
        System.out.print("");
      } catch(ParseException pe) {
	logger.log(Level.SEVERE, "position: " + pe.getPosition());
      }
    } catch (IOException e) {
     logger.log(Level.SEVERE, "URL exception in ensemblAvailableGenomes.");
    }
    return genomes;
  } // ensemblAvailableGenomes

}
