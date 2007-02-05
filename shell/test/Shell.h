class Shell
{
		public:
				Shell();

				const std::string& pwe() const;

				void ce( const std::string& dest );

				void create( const std::string& path );

				void remove( const std::string& path );

				void le ( const std::string& path );

				void le ();
		private:
				std::string expandPath( const std::string& path ) const;
				std::string cwe_;
};
