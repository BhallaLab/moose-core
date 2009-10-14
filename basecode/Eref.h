class Eref
{
	public:
		friend ostream& operator <<( ostream& s, const Eref& e );
		Eref( Element* e, unsigned int index );

		/**
		 * returns the sum of all valid incoming entries
		 */
		double sumBuf( SyncId slot );

		/**
		 * Returns the product of all valid incoming entries
		 * with v. If there are no entries, returns v
		 */
		double prdBuf( SyncId slot, double v );

		/**
		 * Returns the single specified entry
		 */
		double oneBuf( SyncId slot );

		/**
		 * Returns the memory location specified by slot.
		 * Used for sends.
		 */
		double* getBufPtr( SyncId slot );

		/**
		 * Sends a double argument
		 */
		void ssend1( SyncId src, double v );

		/**
		 * Sends two double arguments
		 */
		void ssend2( SyncId src, double v1, double v2 );

		/**
		 * Asynchronous message send.
		 */
		void asend( ConnId conn, FuncId func, const char* arg, 
			unsigned int size ) const;

		/**
		 * Asynchronous send to a specific target.
		 */
		void tsend( ConnId conn, FuncId func, Id target, const char* arg, 
			unsigned int size ) const;

		/**
		 * Returns data entry
		 */
		Data* data();

		/**
		 * Returns Element part
		 */
		Element* element() const {
			return e_;
		}

		/**
		 * Returns index part
		 */
		unsigned int index() const {
			return i_;
		}
	private:
		Element* e_;
		unsigned int i_;
};
