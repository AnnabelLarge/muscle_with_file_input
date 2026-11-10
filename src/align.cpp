#include "muscle.h"
#include "mega.h"

void MakeReplicateFileName(const string &Pattern, TREEPERM TP,
  uint PerturbSeed, string &FileName)
	{
	FileName.clear();

	size_t pos = Pattern.find('@');
	if (pos == string::npos)
		Die("'@' not found in '%s'", Pattern.c_str());

	for (size_t i = 0; i < pos; ++i)
		FileName += Pattern[i];

	Psa(FileName, "%s.%u", TREEPERMToStr(TP), PerturbSeed);

	for (size_t i = pos+1; i < SIZE(Pattern); ++i)
		FileName += Pattern[i];
	}

static void Align(MPCFlat &M, MultiSequence &InputSeqs,
  uint PerturbSeed, TREEPERM TP, bool WriteEfaHdr, FILE *fOut)
	{
	if (fOut == 0)
		return;

	bool Nucleo = (g_Alpha == ALPHA_Nucleo);
	HMMParams HP;
	if (optset_hmmin)
		HP.FromFile(opt(hmmin));
	else
		HP.FromDefaults(Nucleo);
	HP.CmdLineUpdate();
	if (PerturbSeed > 0)
		{
		ResetRand(PerturbSeed);
		HP.PerturbProbs(PerturbSeed);
		}
	HP.ToPairHMM();

	M.m_TreePerm = TP;
	M.Run(&InputSeqs);

	asserta(M.m_MSA != 0);

	if (WriteEfaHdr)
		{
		const char *TPStr = TREEPERMToStr(TP);
		fprintf(fOut, "<%s.%u\n", TPStr, PerturbSeed);
		}
	M.m_MSA->WriteMFA(fOut);
	}

// Options that read or write posterior files (postio.cpp).
static void SetPostOpts(MPCFlat &M, uint InputSeqCount)
	{
	const bool In = optset_postdir || optset_postmap;
	const bool Out = optset_postout;
	if (In && !(optset_postdir && optset_postmap && optset_postfamily))
		Die("-postdir, -postmap and -postfamily must be set together");
	if (optset_postfamily && !In && !Out)
		Die("-postfamily needs -postdir or -postout");
	if (!In && !Out)
		{
		ProgressLog("Posteriors: built-in\n");
		return;
		}

	const char *Opt = (In ? "-postdir" : "-postout");
	if (Mega::m_Loaded)
		Die("%s is not supported with Mega input", Opt);
	if (optset_minsuper && InputSeqCount >= opt(minsuper))
		Die("%s is not supported with Super5 (-minsuper)", Opt);
	if (opt(stratified) || opt(diversified) || optset_replicates)
		Die("%s needs a single run (not -stratified, -diversified or "
		  "-replicates)", Opt);

	if (In)
		{
	// These change only the pair HMM, which is not run with -postdir.
		if (optset_perturb && opt(perturb) > 0)
			Die("-perturb has no effect with -postdir");
		if (optset_hmmin || optset_s_is || optset_s_il || optset_m_is ||
		  optset_m_il || optset_is_is || optset_il_il)
			Die("-hmmin and the HMM transition options have no effect "
			  "with -postdir");
		if (Out && opt(postout) == opt(postdir))
			Die("-postout must differ from -postdir");
		M.m_PostInDir = opt(postdir);
		M.m_PostMapFileName = opt(postmap);
		M.m_PostFamily = opt(postfamily);
		ProgressLog("Posteriors: from %s (map %s, family %s)\n",
		  M.m_PostInDir.c_str(), M.m_PostMapFileName.c_str(),
		  M.m_PostFamily.c_str());
		}
	else
		ProgressLog("Posteriors: built-in\n");

	if (Out)
		{
		M.m_PostOutDir = opt(postout);
		if (optset_postfamily)
			M.m_PostFamily = opt(postfamily);
		ProgressLog("Posteriors saved to %s (family %s)\n",
		  M.m_PostOutDir.c_str(), M.m_PostFamily.c_str());
		}
	}

void cmd_align()
	{
	MultiSequence InputSeqs;
	LoadInput(InputSeqs);
	const uint InputSeqCount = InputSeqs.GetSeqCount();

	MPCFlat M;
	SetPostOpts(M, InputSeqCount);

	if (optset_minsuper && InputSeqCount >= opt(minsuper))
		{
		InputSeqs.Clear();
		Progress("%u seqs, running Super5 algorithm\n", InputSeqCount);
		opt_super5 = opt(align);
		optset_super5 = true;
		void cmd_super5();
		cmd_super5();
		return;
		}

	if (InputSeqCount > 1000)
		Warning(">1k sequences, may be slow or use excessive memory, consider using -super5");

	const string &OutputPattern = opt(output);
	if (OutputPattern.empty())
		Die("Must set -output");

	//ShowSeqStats(InputSeqs);

	bool OutputWildcard = OutputPattern.find('@') != string::npos;
	FILE *fOut = 0;

	bool IsNucleo = InputSeqs.GuessIsNucleo();
	if (IsNucleo)
		SetAlpha(ALPHA_Nucleo);
	else
		SetAlpha(ALPHA_Amino);

	if (optset_consiters)
		M.m_ConsistencyIterCount = opt(consiters);
	if (optset_refineiters)
		M.m_RefineIterCount = opt(refineiters);

	if (opt(stratified) && opt(diversified))
		Die("Cannot set both -stratified and -diversified");

	if (opt(stratified) || opt(diversified))
		{
		if (optset_perm || optset_perturb)
			Die("Cannot set -perm or -perturb with -stratified or -diversified");
		}

	uint RepCount = 1;
	if (opt(stratified))
		RepCount = 4;
	else if (opt(diversified))
		RepCount = 100;

	if (optset_replicates)
		RepCount = opt(replicates);

	if (RepCount == 1)
		{
		uint PerturbSeed = 0;
		if (optset_perturb)
			PerturbSeed = opt(perturb);

		TREEPERM TP = TP_None;
		if (optset_perm)
			TP = StrToTREEPERM(opt(perm));
		if (TP == TP_All)
			Die("-perm all not supported, use -stratified");

		string OutputFileName;
		if (OutputWildcard)
			MakeReplicateFileName(OutputPattern, TP, PerturbSeed, OutputFileName);
		else
			OutputFileName = OutputPattern;
		fOut = CreateStdioFile(OutputFileName);
		Align(M, InputSeqs, PerturbSeed, TP, false, fOut);
		CloseStdioFile(fOut);
		return;
		}

	bool Stratified = false;
	if (opt(stratified))
		{
		Stratified = true;
		RepCount *= 4;
		if (optset_perm)
			Die("Cannot set both -perm and -stratified");
		asserta(RepCount > 0);
		}

	string OutputFileName;
	if (!OutputWildcard)
		fOut = CreateStdioFile(OutputPattern);
	for (uint RepIndex = 0; RepIndex < RepCount; ++RepIndex)
		{
		uint PerturbSeed = (Stratified ? RepIndex/4 : RepIndex);
		TREEPERM TP = (optset_perm ? 
		  StrToTREEPERM(opt(perm)) : TREEPERM(RepIndex%4));
		ProgressLog("Replicate %u/%u, %s.%u\n",
		  RepIndex+1, RepCount, TREEPERMToStr(TP), PerturbSeed);

		if (OutputWildcard)
			{
			MakeReplicateFileName(OutputPattern, TP, PerturbSeed, OutputFileName);
			fOut = CreateStdioFile(OutputFileName);
			}
		bool WriteEfaHeader = !OutputWildcard;
		Align(M, InputSeqs, PerturbSeed, TP, WriteEfaHeader, fOut);
		if (OutputWildcard)
			CloseStdioFile(fOut);
		}

	if (!OutputWildcard)
		CloseStdioFile(fOut);
	}
