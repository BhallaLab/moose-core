#ifndef _Reaction_h
#define _Reaction_h
class Reaction
{
	friend class ReactionWrapper;
	public:
		Reaction()
		{
			kf_ = 0.1;
			kb_ = 0.1;
		}

	private:
		double kf_;
		double kb_;
		double A_;
		double B_;
};
#endif // _Reaction_h
