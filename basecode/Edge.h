
/**
 * The Edge class specifies a directional link between two Erefs.
 * Actual messages ride on the Edge. There may be multiple messages
 * and they may go in either direction.
 */
class Edge // edge
{
	public:
		Edge( Eref src, Eref dest )
			: src_( src ), dest_( dest )
		Eref src();
		Eref dest();
		vector< unsigned int >& srcRange( unsigned int dest );
		vector< unsigned int >& destRange( unsigned int src );
		// bool add( unsigned int srcIndex, unsigned int destIndex );
		// MsgInfo src();
		// MsgInfo dest();

	private:
		Eref src_;
		Eref dest_;
		vector< char > syncBuf;		
};
