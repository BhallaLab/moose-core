/**
 * Encapsulates edges with direction info and functions to associate.
 */
class Msg
{
	public:
	private:
		vector< Edge* > e_; // Edges in this msg.
		vector< bool > isSrc_; // Direction of each edge.
		vector< FuncId > func_; // Operations to do on this edge.

};

class Msg
{
	public:
	private:
		Edge* e_; // Edges in this msg.
		vector< bool > isSrc_; // Direction of each edge.
		vector< FuncId > func_; // Operations to do on this edge.
};


/**
 * Note that bot the src and dest see things as one-to-many. We specially
 * pay attention to the src viewpoint, because we send info that way.
 * Actually now the opposite: We send info to a single point and have
 * to harvest it from many points.
 */
